import pandas as pd
from datasets import Dataset, ClassLabel, Features, Value

cf_url = "https://raw.githubusercontent.com/tdiggelm/climate-fever-dataset/main/dataset/climate-fever.jsonl"
cf_orig = pd.read_json(cf_url, lines=True)
cf_orig.head()

df = (
    cf_orig.set_index(["claim_id"])["evidences"]
    .apply(pd.Series)
    .stack()
    .apply(pd.Series)
    .reset_index()
    .drop("level_1", axis=1)
)

claim_info = cf_orig[["claim_id", "claim", "claim_label"]]
df = claim_info.merge(df, how="left", on="claim_id")

df["label"] = df["evidence_label"]
df_final = df[["claim_id", "claim", "evidence", "label", "article"]]
df_final = df_final.rename({"article": "category"}, axis=1)
# df_final.head()

# convert to dataset and split into test/train/val
ds = Dataset.from_pandas(df_final, preserve_index=False)
print(ds.features)

# start with train/test split
ds = ds.train_test_split(test_size=0.2, seed=727)

# split training into train and validation
train_val_ds = ds["train"].train_test_split(test_size=0.3, seed=451)

# update original ds with re-split training and validation
ds["train"] = train_val_ds["train"]
ds["valid"] = train_val_ds["test"]

# one last step: convert string labels to ClassLabel
# a function to map the string labels to integers:

def label_to_int(example):
    label_to_int_mapping = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT_ENOUGH_INFO': 2}
    example['label'] = label_to_int_mapping[example['label']]
    return example

# apply the mapping function to the dataset
ds = ds.map(label_to_int)

features = Features({
    'claim_id': Value('int64'),
    'claim': Value('string'),
    'evidence': Value('string'),
    'label': ClassLabel(num_classes=3, names=['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']),
    'category': Value('string')
})

def set_features(example):
    return example

ds = ds.map(set_features, features=features)

# # Accessing the integer value of the first example's label
# print(ds['train']['label'][0])
# # Accessing the original string of the first example's label
# print(ds['train'].features['label'].int2str(ds['train']['label'][0]))
# print(ds)

# only need to do this once
ds.push_to_hub("rexarski/climate_fever_fixed", private=False)
