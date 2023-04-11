from openai.error import OpenAIError
import configparser
import os
from utils import (
    embed_docs,
    get_answer,
    get_sources,
    parse_txt,
    search_docs,
    text_to_docs,
)
from tqdm import tqdm

# import json
import pandas as pd
import numpy as np
import os
import glob

config = configparser.ConfigParser()
config.read("config.ini")
os.environ["OPENAI_API_KEY"] = config.get("API", "openai_api_key")
os.environ["OPENAI_MODEL"] = config.get("API", "model_engine")

path = "data/txt"
txt_files = glob.glob(os.path.join(path, "*.txt"))
valid_indices = np.array(
    sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in txt_files])
)

df_tcfd = pd.read_csv("data/tcfd.csv")

df_tcfd = df_tcfd.loc[df_tcfd.index.isin(valid_indices)]
df_tcfd = df_tcfd.reset_index().rename(columns={"index": "file_id"})

df_tcfd = (
    df_tcfd.assign(disclosure=df_tcfd["Recommended Disclosure"].str.split("; "))
    .explode("disclosure")
    .reset_index(drop=True)
)
df_tcfd["disclosure"] = (
    df_tcfd["disclosure"]
    .str.replace(r"\s[abcd]\)$", "", regex=True)
    .str.strip()
)
df_tcfd = (
    df_tcfd.drop(columns=["Recommended Disclosure"])
    .drop_duplicates()
    .reset_index(drop=True)
)
df_tcfd = df_tcfd[["file_id", "disclosure"]]

# print(df_tcfd.disclosure.value_counts())

df_tcfd["disclosure"] = (
    df_tcfd["disclosure"]
    .str.replace(
        "All disclosures",
        "Risk Management,Strategy,Metrics and Targets,Governance",
    )
    .str.split(",")
)
# prep the dataframe with file_id and disclosure only
print(df_tcfd)

last_file_id = None
index = None

fid_list = df_tcfd["file_id"].to_list()
dis_list = df_tcfd["disclosure"].to_list()

for i in tqdm(range(len(fid_list))):
    file_id = fid_list[i]
    disclosure = dis_list[i]

    docs = []

    if file_id == last_file_id:
        skip_embedding = True
    else:
        last_file_id = file_id
        skip_embedding = False
    with open(f"data/txt/{file_id}.txt", "rb") as file:
        doc = None
        if file is not None:
            if file.name.endswith(".txt"):
                doc = parse_txt(file)
            else:
                raise ValueError("File type not supported!")
        docs.append(doc)

        if not skip_embedding:
            text = text_to_docs(docs)
            try:
                print("Indexing document...")
                index = embed_docs(text)
            except OpenAIError as e:
                print(e._message)

        query = disclosure[0]
        sources = search_docs(index, query)
        try:
            answer = get_answer(sources, query)
        except OpenAIError as e:
            print(e._message)
        sources = get_sources(answer, sources)
        df_tcfd["answer"] = answer["output_text"]

df_tcfd["disclosure"] = df_tcfd["disclosure"].apply(lambda x: x[0])
df_tcfd["answer"] = df_tcfd["answer"].str.split("\n")
df_tcfd = df_tcfd.explode("answer").reset_index(drop=True)
df_tcfd["answer"] = (
    df_tcfd["answer"].str.replace(r"^\d+\.\s", "", regex=True).str.strip()
)
df_tcfd.to_csv("data/tcfd_output.csv", index=False)
