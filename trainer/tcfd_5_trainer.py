# !huggingface-cli login

import pandas as pd
import numpy as np
import json

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import requests

import torch

from transformers import EarlyStoppingCallback
from transformers import RobertaTokenizerFast
from transformers import (
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

num_classes = 5

label_to_id = {
    "Governance": 0,
    "Metrics and Targets": 1,
    "Risk Management": 2,
    "Strategy": 3,
    "None": 4,
}

np.random.seed(0)
set_seed(42)


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


texts = []
labels = []

url = "https://raw.githubusercontent.com/ClimateBert/training-example/main/training_data.json"
response = requests.get(url)
training_data = json.loads(response.text)

for sample in training_data:
    texts.append(sample["text"])
    labels.append(label_to_id[sample["label"]])

assert len(texts) == len(labels)

texts = np.array(texts)
labels = np.array(labels)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.05, random_state=42
)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.3, random_state=42
)

print(f"Train samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(test_texts)}")

# Train samples: 34886
# Validation samples: 14952
# Test samples: 2624

tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")


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

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

train_dataset = TCFDDataset(train_encodings, train_labels)
val_dataset = TCFDDataset(val_encodings, val_labels)

training_args = TrainingArguments(
    output_dir="rexarski/distilroberta-tcfd-disclosure-5",  # output directory
    overwrite_output_dir=True,
    push_to_hub=True,
    num_train_epochs=10,  # total number of training epochs
    per_device_train_batch_size=24,  # batch size per device during training
    per_device_eval_batch_size=24,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,
    fp16=True,  # enable mixed precision training if supported by GPU
    gradient_accumulation_steps=4,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

model = RobertaForSequenceClassification.from_pretrained(
    "distilroberta-base", num_labels=num_classes
)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

early_stop = EarlyStoppingCallback(2)

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
            f"{test_texts[i]}\t{test_labels[i]}\t{x[i,0]}\t{x[i,1]}\t{x[i,2]}\t{x[i,3]}\t{x[i,4]}\n "
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
        ]
    ].values,
    axis=1,
)

X = []
y = []

print(f"Test accuracy: {np.mean(df.label == df.pred_class):.4f}")
print(
    f"Test F1 score: {f1_score(df.label, df.pred_class, average='macro'):.4f}"
)

# Test accuracy: 0.8075
# Test F1 score: 0.6268

trainer.push_to_hub()
