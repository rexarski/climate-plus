import pandas as pd
from random import randint
from datasets import load_dataset, Dataset

# from huggingface_hub import notebook_login

# climate_fever (only need to run once)

ds = load_dataset("climate_fever")
# ds
# print("claim:", ds["test"]["claim"][1])
# ds["test"]["evidences"][1]

cf_url = "https://raw.githubusercontent.com/tdiggelm/climate-fever-dataset/main/dataset/climate-fever.jsonl"
cf_orig = pd.read_json(cf_url, lines=True)
cf_orig.head()
df = (
    cf_orig.set_index(["claim_id"])["evidences"]
    .apply(pd.Series)
    .stack()
    .apply(pd.Series)
    .reset_index()
    .drop("level_1", 1)
)

claim_info = cf_orig[["claim_id", "claim", "claim_label"]]
df = claim_info.merge(df, how="left", on="claim_id")

df.head(5)

label_map = {
    "SUPPORTS": "entailment",
    "REFUTES": "contradiction",
    "NOT_ENOUGH_INFO": "neutral",
}
df["label"] = df["evidence_label"].map(label_map)
df.head(10)

df_final = df[["claim_id", "claim", "evidence", "label", "article"]]
df_final = df_final.rename({"article": "category"}, axis=1)
df_final.head()

### Convert to dataset and split into test/train/val

ds = Dataset.from_pandas(df_final, preserve_index=False)
ds.features

# start with train/test split
ds = ds.train_test_split(test_size=0.2, seed=727)

# split training into train and validation
train_val_ds = ds["train"].train_test_split(test_size=0.3, seed=451)

# update original ds with re-split training and validation
ds["train"] = train_val_ds["train"]
ds["valid"] = train_val_ds["test"]

# only need to do this once
# ds.push_to_hub("rexarski/climate_fever_fixed", private=True)
