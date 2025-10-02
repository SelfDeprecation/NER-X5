from typing import List, Dict
from datasets import Dataset


def prepare_labels_mapping(labels_list: List[List[str]]):
    unique = set()
    for row in labels_list:
        unique.update(row)
    unique = sorted(unique)
    label_to_id = {l: i for i, l in enumerate(unique)}
    id_to_label = {i: l for l, i in label_to_id.items()}
    return label_to_id, id_to_label


def build_hf_tokenized_dataset(samples: List[str], annotations: List[List[str]], tokenizer, label_to_id: Dict[str,int], max_length: int = 128, label_all_tokens: bool = False):
    words_list = [s.split(' ') for s in samples]
    enc = tokenizer(words_list, is_split_into_words=True, truncation=True, padding='max_length', max_length=max_length)

    labels_enc = []
    for i, word_labels in enumerate(annotations):
        word_ids = enc.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label = word_labels[word_idx]
                if label_all_tokens:
                    label_ids.append(label_to_id[label])
                else:
                    if word_idx != previous_word_idx:
                        label_ids.append(label_to_id[label])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
        labels_enc.append(label_ids)

    dataset = Dataset.from_dict({**enc, 'labels': labels_enc})
    return dataset
