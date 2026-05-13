import numpy as np
import pandas as pd

def get_default_features():
    # Detection 실패 시에도 XGBoost 입력 차원을 유지하기 위한 fallback feature이다.
    return {
        "art_area": 1,
        "art_major": 1,
        "art_minor": 1,
        "art_ecc": 1,
        "art_axisratio": 1,
        "art_round": 1,
        "ijv_area": 1,
        "ijv_major": 1,
        "ijv_minor": 1,
        "ijv_ecc": 1,
        "ijv_axisratio": 1,
        "ijv_round": 1,
    }

def features_to_array(B2):
    # dict feature를 scaler와 XGBoost가 기대하는 DataFrame 형태로 변환한다.
    columns = [
        "art_area",
        "art_major",
        "art_minor",
        "art_ecc",
        "art_axisratio",
        "art_round",
        "ijv_area",
        "ijv_major",
        "ijv_minor",
        "ijv_ecc",
        "ijv_axisratio",
        "ijv_round",
    ]

    feature_dict = {
        "art_area": float(B2["art_area"]),
        "art_major": float(B2["art_major"]),
        "art_minor": float(B2["art_minor"]),
        "art_ecc": float(B2["art_ecc"]),
        "art_axisratio": float(B2["art_axisratio"]),
        "art_round": float(B2["art_round"]),

        "ijv_area": float(B2["ijv_area"]),
        "ijv_major": float(B2["ijv_major"]),
        "ijv_minor": float(B2["ijv_minor"]),
        "ijv_ecc": float(B2["ijv_ecc"]),
        "ijv_axisratio": float(B2["ijv_axisratio"]),
        "ijv_round": float(B2["ijv_round"]),
    }

    X = pd.DataFrame([feature_dict])

    # scaler 학습 때 사용한 column 순서와 동일하게 맞춰 feature mismatch를 방지한다.
    X = X[columns]

    return X

def calculate_vessel_features(final_mask):
    # 현재 함수는 area 기반 기본 feature 확인용이며, 실제 ROSC 분류 feature는 classifier 내부에서 계산된다.
    artery_mask = (final_mask == 1).astype(np.uint8)
    vein_mask = (final_mask == 2).astype(np.uint8)

    artery_area = int(artery_mask.sum())
    vein_area = int(vein_mask.sum())

    return {
        "artery_area": artery_area,
        "vein_area": vein_area
    }

def prob_to_logit(p, eps=1e-6):
    # probability를 logit으로 바꿀 때 0 또는 1에서 overflow가 나지 않도록 clipping한다.
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def logit_to_prob(logit):
    return 1 / (1 + np.exp(-logit))