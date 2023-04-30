from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset
from collections import Counter

model = AutoModelForSequenceClassification.from_pretrained(
    "rexarski/bert-base-climate-fever-fixed"
)
tokenizer = AutoTokenizer.from_pretrained(
    "rexarski/bert-base-climate-fever-fixed"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = tokenizer(
    ["says Sweet, who has authored several sea-level rise studies."],
    [
        "Over the 21st century, the IPCC projects that in a very high emissions scenario the sea level could rise by 61–110 cm."
    ],
    padding="max_length",
    truncation=True,
    return_tensors="pt",
    max_length=512,
)

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    label_mapping = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
    print(labels)

# ds = load_dataset("rexarski/climate_fever_fixed")["test"]
# print("Value counts: ", Counter(ds["label"]))
# # Value counts:  Counter({2: 996, 0: 375, 1: 164})
# # approxiamtely 64% of the dataset is NOT_ENOUGH_INFO
