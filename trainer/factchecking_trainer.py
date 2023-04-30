from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import time
from tqdm import tqdm
from sklearn.metrics import f1_score

# Run `huggingface-cli login` to login to your HuggingFace account

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

base_model = "bert-base-uncased"
# model_checkpoint = 'climatebert/distilroberta-base-climate-f'
ds = load_dataset("rexarski/climate_fever_fixed")


model = AutoModelForSequenceClassification.from_pretrained(
    base_model, num_labels=3
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
model.to(device)

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

num_added_toks = tokenizer.add_tokens(special_tokens)

# print(f"Number of tokens added: {num_added_toks}")


class climate_fever_f_bert(Dataset):
    def __init__(self, ds, base_model):
        self.label_dict = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}

        self.train_df = ds["train"]
        self.val_df = ds["valid"]
        self.test_df = ds["test"]

        # pretrained base model tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, do_lower_case=True
        )
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.init_data()

    def init_data(self):
        self.train_data = self.load_data(self.train_df)
        self.val_data = self.load_data(self.val_df)
        self.test_data = self.load_data(self.test_df)

    def load_data(self, df):
        MAX_LEN = 512
        token_ids = []
        mask_ids = []
        seg_ids = []
        y = []

        claim_list = df["claim"]
        evidence_list = df["evidence"]
        label_list = df["label"]

        for claim, evidence, label in zip(
            claim_list, evidence_list, label_list
        ):
            claim_id = self.tokenizer.encode(
                claim,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_LEN,
            )
            evidence_id = self.tokenizer.encode(
                evidence,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_LEN,
            )
            pair_token_ids = (
                [self.tokenizer.cls_token_id]
                + claim_id
                + [self.tokenizer.sep_token_id]
                + evidence_id
                + [self.tokenizer.sep_token_id]
            )

            claim_len = len(claim_id)
            evidence_len = len(evidence_id)

            segment_ids = torch.tensor(
                [0] * (claim_len + 2) + [1] * (evidence_len + 1)
            )  # sentence 0 and sentence 1
            attention_mask_ids = torch.tensor(
                [1] * (claim_len + evidence_len + 3)
            )  # mask padded values

            token_ids.append(torch.tensor(pair_token_ids))
            seg_ids.append(segment_ids)
            mask_ids.append(attention_mask_ids)
            # y.append(self.label_dict[label])
            y.append(label)

        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
        return dataset

    def get_data_loaders(self, batch_size=32, shuffle=True):
        train_loader = DataLoader(
            self.train_data,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=True,
        )

        val_loader = DataLoader(
            self.val_data,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=True,
        )

        test_loader = DataLoader(
            self.val_data,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=True,
        )

        return train_loader, val_loader, test_loader


climate_dataset = climate_fever_f_bert(ds, base_model)
train_loader, val_loader, test_loader = climate_dataset.get_data_loaders(
    batch_size=8
)

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "gamma", "beta"]
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay_rate": 0.01,
    },
    {
        "params": [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ],
        "weight_decay_rate": 0.0,
    },
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)


def multi_acc(y_pred, y_test):
    acc = (
        torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test
    ).sum().float() / float(y_test.size(0))
    return acc


def multi_f1(y_pred, y_test):
    y_pred = torch.log_softmax(y_pred, dim=1).argmax(dim=1).cpu().numpy()
    y_test = y_test.cpu().numpy()
    f1 = f1_score(y_test, y_pred, average="macro")
    return f1


EPOCHS = 10


def train(model, train_loader, val_loader, optimizer):
    total_step = len(train_loader)
    best_val_acc = 0
    best_val_f1 = 0

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()

        total_train_acc = 0
        total_train_loss = 0
        total_train_f1 = 0

        for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in tqdm(
            enumerate(train_loader)
        ):
            optimizer.zero_grad()
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            labels = y.to(device)

            loss, prediction = model(
                pair_token_ids,
                token_type_ids=seg_ids,
                attention_mask=mask_ids,
                labels=labels,
            ).values()

            acc = multi_acc(prediction, labels)
            f1score = multi_f1(prediction, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += acc.item()
            total_train_f1 += f1score

        train_acc = total_train_acc / len(train_loader)
        train_loss = total_train_loss / len(train_loader)
        train_f1 = total_train_f1 / len(train_loader)

        model.eval()

        total_val_acc = 0
        total_val_loss = 0
        total_val_f1 = 0

        with torch.no_grad():
            for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(
                val_loader
            ):
                optimizer.zero_grad()
                pair_token_ids = pair_token_ids.to(device)
                mask_ids = mask_ids.to(device)
                seg_ids = seg_ids.to(device)
                labels = y.to(device)

                loss, prediction = model(
                    pair_token_ids,
                    token_type_ids=seg_ids,
                    attention_mask=mask_ids,
                    labels=labels,
                ).values()

                acc = multi_acc(prediction, labels)
                f1score = multi_f1(prediction, labels)

                total_val_loss += loss.item()
                total_val_acc += acc.item()
                total_val_f1 += f1score

        val_acc = total_val_acc / len(val_loader)
        val_f1 = total_val_f1 / len(val_loader)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_f1_model.pth")
        val_loss = total_val_loss / len(val_loader)

        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)

        print(
            f"Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} train_f1: {train_f1:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} val_f1: {val_f1:.4f}"
        )
        print(
            "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        )


train(model, train_loader, val_loader, optimizer)

# Pick the best model and push it to huggingface

# load the saved state dict
state_dict = torch.load("best_f1_model.pth")

# assign the state dict to the model
model.load_state_dict(state_dict)

training_args = TrainingArguments(
    output_dir="rexarski/bert-base-climate-fever-fixed", push_to_hub=True
)
trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)

trainer.push_to_hub()

total_test_acc = 0
total_test_f1 = 0

with torch.no_grad():
    for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(
        test_loader
    ):
        optimizer.zero_grad()
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = y.to(device)

        loss, prediction = model(
            pair_token_ids,
            token_type_ids=seg_ids,
            attention_mask=mask_ids,
            labels=labels,
        ).values()

        acc = multi_acc(prediction, labels)
        f1score = multi_f1(prediction, labels)

        total_test_acc += acc.item()
        total_test_f1 += f1score.item()

test_acc = total_test_acc / len(test_loader)
test_f1 = total_test_f1 / len(test_loader)

print(f"The accuracy of testing split is: {test_acc:.4f}")
print(f"The f1 score of testing split is: {test_f1:.4f}")
