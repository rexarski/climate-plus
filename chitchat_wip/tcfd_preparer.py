from openai.error import OpenAIError
import configparser
import os
from utils import (
    embed_docs,
    get_answer,
    parse_pdf,
    search_docs,
    text_to_docs,
)
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import glob

config = configparser.ConfigParser()
config.read("config.ini")
os.environ["OPENAI_API_KEY"] = config.get("API", "openai_api_key")
os.environ["OPENAI_MODEL"] = config.get("API", "model_engine")

path = "data/pdf"
pdf_files = glob.glob(os.path.join(path, "*.pdf"))
valid_indices = np.array(
    sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in pdf_files])
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
df_tcfd["disclosure"] = df_tcfd["disclosure"].str.replace(
    "All disclosures",
    "Risk Management,Strategy,Metrics and Targets,Governance",
)
df_tcfd = (
    df_tcfd.assign(disclosure=df_tcfd["disclosure"].str.split(","))
    .explode("disclosure")
    .reset_index(drop=True)
)
# prep the dataframe with file_id and disclosure only
fid_list = df_tcfd["file_id"].to_list()
dis_list = df_tcfd["disclosure"].to_list()
flist = []
dlist = []
answer_list = []
for i in tqdm(range(len(fid_list))):
    file_id = fid_list[i]
    disclosure = dis_list[i]
    docs = []
    with open(f"data/pdf/{file_id}.pdf", "rb") as file:
        doc = None
        if file is not None:
            if file.name.endswith(".pdf"):
                try:
                    doc = parse_pdf(file)
                except TypeError:
                    print(
                        f"data/pdf/{file_id}.pdf has TypeError! need some investigation"
                    )
                    continue
            else:
                raise ValueError("File type not supported!")
        docs.append(doc)
        text = text_to_docs(docs)
        try:
            index = embed_docs(text)
        except OpenAIError as e:
            print(e._message)
        query = disclosure
        sources = search_docs(index, query)
        try:
            answer = get_answer(sources, query)
        except OpenAIError as e:
            print(e._message)
        temp_list = answer["output_text"].split("\n")
        flist.extend([file_id] * len(temp_list))
        dlist.extend([disclosure] * len(temp_list))
        answer_list.extend(temp_list)
        print(f"{file_id} {disclosure} is completed.")
        if len(answer_list) == len(flist) == len(dlist):
            print("All good!")
df = pd.DataFrame({"text": answer_list, "label": dlist})
df.to_csv("data/tcfd_output.csv", index=False)
