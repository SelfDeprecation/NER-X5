import numpy as np
from seqeval.metrics import classification_report, f1_score


def align_predictions(predictions, label_ids, id_to_label):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    out_labels = []
    for i in range(batch_size):
        pred_labels = []
        true_labels = []
        for j in range(seq_len):
            if label_ids[i, j] != -100:
                pred_labels.append(id_to_label[preds[i, j]])
                true_labels.append(id_to_label[label_ids[i, j]])
        out_labels.append((pred_labels, true_labels))
    return out_labels


def compute_metrics(p):
    preds = p.predictions
    label_ids = p.label_ids
    id_to_label = p.id_to_label
    aligned = align_predictions(preds, label_ids, id_to_label)
    y_true = [t for _, t in aligned]
    y_pred = [p for p, _ in aligned]
    report = classification_report(y_true, y_pred, digits=4)
    macro = f1_score(y_true, y_pred, average='macro')
    return {'macro_f1': macro, 'report': report}
