# !huggingface-cli login

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch

from transformers import EarlyStoppingCallback
from transformers import RobertaTokenizerFast
from transformers import (
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from datasets import load_dataset

num_classes = 11
set_seed(42)
ds = load_dataset("rexarski/TCFD_disclosure")

label_to_id = {
    "Governance a)": 0,
    "Governance b)": 1,
    "Metrics and Targets a)": 2,
    "Metrics and Targets b)": 3,
    "Metrics and Targets c)": 4,
    "Risk Management a)": 5,
    "Risk Management b)": 6,
    "Risk Management c)": 7,
    "Strategy a)": 8,
    "Strategy b)": 9,
    "Strategy c)": 10,
}

texts = np.array(ds["train"]["text"])
labels = [label_to_id[x] for x in ds["train"]["label"]]
labels = np.array(labels)

assert len(texts) == len(labels)  # 593

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.05, random_state=42
)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.3, random_state=42
)

print(f"Train samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(test_texts)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")

print(f"The original tokenizer size is {len(tokenizer)}")

special_tokens = [
    "CO2",
    "emissions",
    "temperature",
    "environmental",
    "soil",
    "increase",
    "conditions",
    "potential",
    "increased",
    "areas",
    "degrees",
    "across",
    "systems",
    "emission",
    "precipitation",
    "impacts",
    "compared",
    "countries",
    "sustainable",
    "provide",
    "reduction",
    "annual",
    "reduce",
    "greenhouse",
    "approach",
    "processes",
    "factors",
    "observed",
    "renewable",
    "temperatures",
    "distribution",
    "studies",
    "variability",
    "significantly",
    "–",
    "further",
    "regions",
    "addition",
    "showed",
    '"',
    "industry",
    "consumption",
    "regional",
    "risks",
    "atmospheric",
    "supply",
    "companies",
    "plants",
    "biomass",
    "electricity",
    "respectively",
    "activities",
    "communities",
    "climatic",
    "solar",
    "investment",
    "spatial",
    "rainfall",
    "•",
    "sustainability",
    "costs",
    "reduced",
    "2021",
    "influence",
    "vegetation",
    "sources",
    "possible",
    "ecosystem",
    "scenarios",
    "summer",
    "drought",
    "structure",
    "economy",
    "considered",
    "various",
    "atmosphere",
    "several",
    "technologies",
    "transition",
    "assessment",
    "dioxide",
    "ocean",
    "fossil",
    "patterns",
    "waste",
    "solutions",
    "transport",
    "strategy",
    "CH4",
    "policies",
    "understanding",
    "concentration",
    "customers",
    "methane",
    "applied",
    "increases",
    "estimated",
    "flood",
    "measured",
    "thermal",
    "concentrations",
    "decrease",
    "greater",
    "following",
    "proposed",
    "trends",
    "basis",
    "provides",
    "operations",
    "differences",
    "hydrogen",
    "adaptation",
    "methods",
    "capture",
    "variation",
    "reducing",
    "N2O",
    "parameters",
    "ecosystems",
    "investigated",
    "yield",
    "strategies",
    "indicate",
    "caused",
    "dynamics",
    "obtained",
    "efforts",
    "coastal",
    "become",
    "agricultural",
    "decreased",
    "GHG",
    "materials",
    "mainly",
    "relationship",
    "+/-",
    "challenges",
    "nitrogen",
    "forests",
    "trend",
    "estimates",
    "towards",
    "Committee",
    "seasonal",
    "developing",
    "particular",
    "importance",
    "tropical",
    "ratio",
    "2030",
    "composition",
    "employees",
    "characteristics",
    "scenario",
    "measurements",
    "plans",
    "fuels",
    "infrastructure",
    "overall",
    "responses",
    "presented",
    "least",
    "assess",
    "diversity",
    "periods",
    "delta",
    "included",
    "already",
    "targets",
    "achieve",
    "affect",
    "conducted",
    "operating",
    "populations",
    "variations",
    "studied",
    "additional",
    "construction",
    "northern",
    "variables",
    "soils",
    "ensure",
    "recovery",
    "combined",
    "decision",
    "practices",
    "however",
    "determined",
    "resulting",
    "mitigation",
    "conservation",
    "estimate",
    "identify",
    "observations",
    "losses",
    "productivity",
    "agreement",
    "monitoring",
    "investments",
    "pollution",
    "contribution",
    "opportunities",
    "simulations",
    "gases",
    "statements",
    "planning",
    "shares",
    "sediment",
    "flux",
    "requirements",
    "trees",
    "temporal",
    "determine",
    "southern",
    "previous",
    "integrated",
    "relatively",
    "analyses",
    "means",
    "2050",
    '"',
    "uncertainty",
    "pandemic",
    "fluxes",
    "findings",
    "moisture",
    "consistent",
    "decades",
    "snow",
    "performed",
    "contribute",
    "crisis",
]

tokenizer.add_tokens(special_tokens)

print(f"The new tokenizer siez is {len(tokenizer)}")

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)


class TCFDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = TCFDDataset(train_encodings, train_labels)
val_dataset = TCFDDataset(val_encodings, val_labels)

training_args = TrainingArguments(
    output_dir="rexarski/distilroberta-tcfd-disclosure",  # output directory
    overwrite_output_dir=True,
    push_to_hub=True,
    num_train_epochs=20,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=8,  # batch size for evaluation
    warmup_steps=50,  # number of warmup steps for learning rate scheduler
    weight_decay=0.02,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,
    fp16=True,  # enable mixed precision training if supported by GPU
    gradient_accumulation_steps=5,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

model = RobertaForSequenceClassification.from_pretrained(
    "distilroberta-base", num_labels=num_classes
)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

early_stop = EarlyStoppingCallback(3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[early_stop],
)

trainer.train()

test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)
test_dataset = TCFDDataset(test_encodings, test_labels)

x = trainer.predict(test_dataset)[0]

with open("output.txt", "a", encoding="utf-8") as fd:
    for i, sent in enumerate(test_dataset):
        fd.write(
            f"{test_texts[i]}\t{test_labels[i]}\t{x[i,0]}\t{x[i,1]}\t{x[i,2]}\t{x[i,3]}\t{x[i,4]}\t{x[i,5]}\t{x[i,6]}\t{x[i,7]}\t{x[i,8]}\t{x[i,9]}\t{x[i,10]}\n "
        )


df = pd.read_csv(
    "output.txt",
    sep="\t",
    header=None,
    names=[
        "text",
        "label",
        "pred_0",
        "pred_1",
        "pred_2",
        "pred_3",
        "pred_4",
        "pred_5",
        "pred_6",
        "pred_7",
        "pred_8",
        "pred_9",
        "pred_10",
    ],
)

df["pred_class"] = np.argmax(
    df[
        [
            "pred_0",
            "pred_1",
            "pred_2",
            "pred_3",
            "pred_4",
            "pred_5",
            "pred_6",
            "pred_7",
            "pred_8",
            "pred_9",
            "pred_10",
        ]
    ].values,
    axis=1,
)

X = []
y = []

id_to_label = {
    0: "Governance",
    1: "Governance",
    2: "Metrics and Targets",
    3: "Metrics and Targets",
    4: "Metrics and Targets",
    5: "Risk Management",
    6: "Risk Management",
    7: "Risk Management",
    8: "Strategy",
    9: "Strategy",
    10: "Strategy",
}

df["label_loose"] = df["label"].map(id_to_label)
df["pred_class_loose"] = df["pred_class"].map(id_to_label)

print(f"Test accuracy: {np.mean(df.label == df.pred_class):.4f}")
print(
    f"Test accuracy (loose): {np.mean(df.label_loose == df.pred_class_loose):.4f}"
)

print(
    f"Test F1 score: {f1_score(df.label, df.pred_class, average='weighted'):.4f}"
)
print(
    f"Test F1 score (loose): {f1_score(df.label_loose, df.pred_class_loose, average='macro'):.4f}"
)
