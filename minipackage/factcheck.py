from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "rexarski/bert-base-climate-fever-fixed"
)
tokenizer = AutoTokenizer.from_pretrained(
    "rexarski/bert-base-climate-fever-fixed"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_mapping = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]


def factcheck(text1, text2):
    features = tokenizer(
        [text1],
        [text2],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        labels = [
            label_mapping[score_max] for score_max in scores.argmax(dim=1)
        ]
    return labels
