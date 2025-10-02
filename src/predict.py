import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def predict_sample(sample: str, tokenizer, model, device):
    words = sample.split(" ")

    enc = tokenizer([words], is_split_into_words=True, truncation=True, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()

    preds = np.argmax(logits, axis=-1)

    label_map = {'1': 'BRAND', '2': 'O', '3': 'PERCENT', '4': 'TYPE', '5': 'VOLUME'}

    word_ids = enc.word_ids(batch_index=0)
    prev_word_idx = None
    word_labels = []
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != prev_word_idx:
            lab = label_map[str(preds[0, idx])]
            word_labels.append(lab)
        prev_word_idx = word_idx

    result = []
    current_pos = 0
    seen_labels = set()
    for i, word in enumerate(words):
        start = current_pos
        end = start + len(word)

        label = word_labels[i] if i < len(word_labels) else "O"
        if label == "O":
            final_label = "O"
        else:
            if label not in seen_labels:
                final_label = f"B-{label}"
                seen_labels.add(label)
            else:
                final_label = f"I-{label}"

        result.append({"start_index": start, "end_index": end, "entity": final_label})
        current_pos = end + 1

    return result
