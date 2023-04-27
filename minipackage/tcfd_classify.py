from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

model = AutoModelForSequenceClassification.from_pretrained(
    "rexarski/distilroberta-tcfd-disclosure"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_mapping = [
    "Governance a)",
    "Governance b)",
    "Metrics and Targets a)",
    "Metrics and Targets b)",
    "Metrics and Targets c)",
    "Risk Management a)",
    "Risk Management b)",
    "Risk Management c)",
    "Strategy a)",
    "Strategy b)",
    "Strategy c)",
]


def tcfd_classify(text):
    features = tokenizer(
        text,
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
