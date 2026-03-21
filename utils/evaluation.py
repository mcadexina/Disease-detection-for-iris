import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_score, recall_score,
    accuracy_score, confusion_matrix, f1_score
)


def compute_far_frr_at_threshold(y_true_binary, y_scores, threshold):
    """Compute FAR and FRR for a binary (one-vs-rest) problem at a given threshold."""
    predictions = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true_binary, predictions, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = cm[0, 0], 0, 0, cm[0, 0]

    # FAR: proportion of non-genuine accepted (impostor accepted)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # FRR: proportion of genuine rejected
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return float(far), float(frr)


def compute_eer(y_true_binary, y_scores):
    """
    Compute Equal Error Rate (EER) — the point where FAR == FRR.
    Returns (eer, far_at_eer, frr_at_eer, threshold_at_eer).
    """
    thresholds = np.linspace(0.0, 1.0, 300)
    fars, frrs = [], []
    for t in thresholds:
        far, frr = compute_far_frr_at_threshold(y_true_binary, y_scores, t)
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)
    diff = np.abs(fars - frrs)
    idx = int(np.argmin(diff))
    eer = float((fars[idx] + frrs[idx]) / 2.0)
    return eer, float(fars[idx]), float(frrs[idx]), float(thresholds[idx])


def compute_multiclass_metrics(y_true, y_pred, y_scores, class_names):
    """
    Compute comprehensive metrics for a multi-class classification problem.

    Args:
        y_true      : 1-D array of true integer class labels
        y_pred      : 1-D array of predicted integer class labels
        y_scores    : 2-D array of shape (n_samples, n_classes) with probabilities
        class_names : list of class name strings aligned with label indices

    Returns:
        dict with overall metrics and per-class FAR/FRR/EER/ROC-AUC data.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_scores = np.asarray(y_scores)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    recall = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    per_class = {}
    sum_far = sum_frr = sum_eer = 0.0

    for i, name in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        scores_i = y_scores[:, i] if y_scores.ndim == 2 else y_scores

        eer, far_eer, frr_eer, eer_thresh = compute_eer(y_bin, scores_i)

        fpr, tpr, _ = roc_curve(y_bin, scores_i)
        roc_auc = float(auc(fpr, tpr))

        per_class[name] = {
            'far': round(far_eer, 4),
            'frr': round(frr_eer, 4),
            'eer': round(eer, 4),
            'roc_auc': round(roc_auc, 4),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'eer_threshold': round(eer_thresh, 4),
        }
        sum_far += far_eer
        sum_frr += frr_eer
        sum_eer += eer

    n = len(class_names)
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'far': round(sum_far / n, 4),
        'frr': round(sum_frr / n, 4),
        'eer': round(sum_eer / n, 4),
        'per_class': per_class,
    }


def plot_roc_curves(metrics, class_names, title="ROC Curves"):
    """
    Return a matplotlib Figure with one ROC curve per class (one-vs-rest).
    metrics must be the dict returned by compute_multiclass_metrics().
    """
    colors = ['#1565C0', '#B71C1C', '#1B5E20', '#E65100']
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, name in enumerate(class_names):
        data = metrics.get('per_class', {}).get(name)
        if data is None:
            continue
        fpr = np.array(data['fpr'])
        tpr = np.array(data['tpr'])
        roc_auc = data['roc_auc']
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Return a matplotlib Figure with the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=35, ha='right')
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, str(cm[r, c]), ha='center', va='center',
                    color='white' if cm[r, c] > thresh else 'black', fontsize=11)

    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    return fig


def plot_far_frr_curve(y_true_binary, y_scores, class_name="Class"):
    """Return a figure showing FAR and FRR vs. threshold and the EER point."""
    thresholds = np.linspace(0.0, 1.0, 300)
    fars, frrs = [], []
    for t in thresholds:
        far, frr = compute_far_frr_at_threshold(y_true_binary, y_scores, t)
        fars.append(far)
        frrs.append(frr)
    fars, frrs = np.array(fars), np.array(frrs)

    eer, far_eer, frr_eer, t_eer = compute_eer(y_true_binary, y_scores)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, fars, label='FAR', color='#B71C1C', lw=2)
    ax.plot(thresholds, frrs, label='FRR', color='#1565C0', lw=2)
    ax.axvline(t_eer, color='grey', linestyle='--', lw=1.5, alpha=0.7)
    ax.scatter([t_eer], [eer], color='black', zorder=5,
               label=f'EER = {eer:.4f}  (t={t_eer:.3f})')
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Rate', fontsize=11)
    ax.set_title(f'FAR / FRR Curve — {class_name}', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig
