import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve


def get_cm_for_class(cm, k):
    TP = TN = FP = FN = 0

    TP = cm[k, k]

    FP = cm[:, k].sum() - TP

    FN = cm[k, :].sum() - TP

    TN = cm.sum() - TP - FP - FN

    return TN, FP, FN, TP


def get_specificity(cm):
    if cm.shape == (2, 2):
        # Binary case
        TN, FP, FN, TP = cm.ravel()
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    elif len(cm.shape) == 3:
        # Multilabel case
        specs = []
        for i in range(len(cm)):
            TN, FP, FN, TP = cm[i].ravel()
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0

            specs.append(spec)

        specificity = np.average(specs)
    else:
        # Multilabel case
        specs = []
        for i in range(len(cm)):
            TN, FP, FN, TP = get_cm_for_class(cm, i)
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0

            specs.append(spec)

        specificity = np.average(specs)

    return specificity


def get_sensitivity(cm):
    if cm.shape == (2, 2):
        # Binary case
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    elif len(cm.shape) == 3:
        # Multilabel case
        senses = []
        for i in range(len(cm)):
            TN, FP, FN, TP = cm[i].ravel()
            sens = TP / (TP + FN) if (TP + FN) > 0 else 0

            senses.append(sens)

        sensitivity = np.average(senses)
    else:
        # Multiclass case
        senses = []
        for i in range(len(cm)):
            TN, FP, FN, TP = get_cm_for_class(cm, i)
            sens = TP / (TP + FN) if (TP + FN) > 0 else 0

            senses.append(sens)

        sensitivity = np.average(senses)

    return sensitivity


def get_f1_score(cm):
    if cm.shape == (2, 2):
        # Binary case
        TN, FP, FN, TP = cm.ravel()
        f1 = TP / (TP + 0.5 * (FP + FN)) if (TP + 0.5 * (FP + FN)) > 0 else 0
    elif len(cm.shape) == 3:
        # Multilabel case
        f1s = []
        for i in range(len(cm)):
            TN, FP, FN, TP = cm[i].ravel()
            f = TP / (TP + 0.5 * (FP + FN)) if (TP + 0.5 * (FP + FN)) > 0 else 0

            f1s.append(f)

        f1 = np.average(f1s)
    else:
        # Multiclass case
        f1s = []
        for i in range(len(cm)):
            TN, FP, FN, TP = get_cm_for_class(cm, i)
            f = TP / (TP + 0.5 * (FP + FN)) if (TP + 0.5 * (FP + FN)) > 0 else 0

            f1s.append(f)

        f1 = np.average(f1s)

    return f1


def get_accuracy(cm):
    if len(cm.shape) == 3:
        # Multilabel case
        accs = []
        for i in range(len(cm)):
            TN, FP, FN, TP = cm[i].ravel()
            total = TN + FP + FN + TP
            acc = (TP + TN) / total if total > 0 else 0
            accs.append(acc)

        accuracy = np.average(accs)
    else:
        # Multiclass and binary case
        correct = np.trace(cm)
        total = np.sum(cm)
        accuracy = correct/total

    return accuracy


def cosine_alignment_loss(image_features, text_features, gt_labels, multilabel):
    # Both image and text features should be normalised
    if multilabel:
        # batch_labels: (batch_size, num_classes), binary
        label_text_features = []
        for i in range(gt_labels.size(0)):
            pos_classes = torch.where(gt_labels[i] > 0)[0]
            if len(pos_classes) == 0:
                # You could just pick a random class but probably would not do what you want it to do
                # pos_class = torch.randint(0, text_features.size(0), (1,))
                raise Exception("You should not be using cosine alignment with ordinal classes switched on.")
            else:
                pos_class = pos_classes[torch.randint(0, len(pos_classes), (1,))]
            label_text_features.append(text_features[pos_class])
        label_text_features = torch.cat(label_text_features, dim=0)
    else:
        label_text_features = text_features[gt_labels]

    cos_sim = F.cosine_similarity(image_features, label_text_features)

    return -torch.mean(cos_sim)


def specificity_at_sensitivity(y_true, y_prob, min_sens=0.9):
    if np.isnan(y_true).any() or np.isnan(y_prob).any() or np.isinf(y_true).any() or np.isinf(y_prob).any():
        print("NaNs or infs found in y_true or y_prob")
        print("y_true:", y_true)
        print("y_prob:", y_prob)
        return 0.0, 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    mask = tpr >= min_sens
    if not np.any(mask):
        return 0.0, 1.0
    idx = np.where(mask)[0][0]
    specificity = 1 - fpr[idx]
    threshold = thresholds[idx]
    return specificity, threshold


def multilabel_specificity_at_sensitivity(y_true, y_prob, min_sens=0.9):
    specs = []
    thresholds = []
    for i in range(y_true.shape[1]):
        spec, thresh = specificity_at_sensitivity(y_true[:, i], y_prob[:, i], min_sens)
        specs.append(spec)
        thresholds.append(thresh)
    return np.mean(specs), np.array(thresholds)


def multiclass_specificity_at_sensitivity(y_true, y_prob, min_sens=0.9):
    specs = []
    thresholds = []
    for i in range(y_prob.shape[1]):
        y_true_bin = (y_true == i).astype(int)
        y_prob_bin = y_prob[:, i]
        spec, thresh = specificity_at_sensitivity(y_true_bin, y_prob_bin, min_sens)
        specs.append(spec)
        thresholds.append(thresh)
    return np.mean(specs), np.array(thresholds)
