import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

def specificity_score(y_true, y_pred):
    """ Arrest를 negative class로 보고 specificity를 계산.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp + 1e-8)

def evaluate_patient_level(csv_path, save_path, threshold=0.5):
    """ 
    true label이 포함된 patient 결과 파일에서만 evaluation을 수행.
    """ 
    df = pd.read_csv(csv_path)

    if "true" not in df.columns:
        print("[SKIP EVAL] patient result csv에 true 컬럼이 없어 evaluation은 생략합니다.")
        return

    y_true = df["true"].astype(int).values
    y_prob = df["prob"].astype(float).values
    y_pred = (y_prob >= threshold).astype(int)

    rows = [
        {"metric": "accuracy", "value": accuracy_score(y_true, y_pred)},
        {"metric": "sensitivity", "value": recall_score(y_true, y_pred, pos_label=1, zero_division=0)},
        {"metric": "specificity", "value": specificity_score(y_true, y_pred)},
        {"metric": "precision", "value": precision_score(y_true, y_pred, pos_label=1, zero_division=0)},
        {"metric": "f1_score", "value": f1_score(y_true, y_pred, pos_label=1, zero_division=0)},
    ]

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    rows.append({"metric": "auc", "value": auc})

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    rows.extend([
        {"metric": "TP", "value": tp},
        {"metric": "FP", "value": fp},
        {"metric": "TN", "value": tn},
        {"metric": "FN", "value": fn},
    ])

    result_df = pd.DataFrame(rows)
    result_df.to_csv(save_path, index=False)

    print("\n===== Patient-level Evaluation =====")
    print(result_df)
    print(f"Saved eval: {save_path}")