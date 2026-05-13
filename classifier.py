import cv2
import numpy as np
import pandas as pd

def extract_ellipse(mask):
    feats = []
    for label in [1, 2]:  # artery=1, IJV=2
        comp = (mask == label).astype(np.uint8)
        area = comp.sum()

        if area < 5:
            feats += [0]*6
            continue

        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)

        if len(c) < 5:
            feats += [0]*6
            continue

        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        major = max(MA, ma)
        minor = min(MA, ma)

        eccentricity = np.sqrt(1 - (minor/major)**2)
        axis_ratio = major / (minor + 1e-8)

        peri = cv2.arcLength(c, True)
        roundness = 4*np.pi*area / (peri**2 + 1e-8)

        feats += [area, major, minor, eccentricity, axis_ratio, roundness]

    return feats

def extract_B2_features(mask):
    feats = extract_ellipse(mask)

    B2 = {
        "art_area": feats[0],
        "art_major": feats[1],
        "art_minor": feats[2],
        "art_ecc": feats[3],
        "art_axisratio":feats[4],
        "art_round": feats[5],
        "ijv_area": feats[6],
        "ijv_major": feats[7],
        "ijv_minor": feats[8],
        "ijv_ecc": feats[9],
        "ijv_axisratio":feats[10],
        "ijv_round": feats[11],
    }

    return B2

def compute_rosc_probability(mask, xgb, scaler):
    feats = extract_B2_features(mask)

    columns = [
        "art_area", "art_major", "art_minor",
        "art_ecc", "art_axisratio", "art_round",
        "ijv_area", "ijv_major", "ijv_minor",
        "ijv_ecc", "ijv_axisratio", "ijv_round",
    ]

    X = pd.DataFrame([{
        "art_area": feats["art_area"],
        "art_major": feats["art_major"],
        "art_minor": feats["art_minor"],
        "art_ecc": feats["art_ecc"],
        "art_axisratio": feats["art_axisratio"],
        "art_round": feats["art_round"],
        "ijv_area": feats["ijv_area"],
        "ijv_major": feats["ijv_major"],
        "ijv_minor": feats["ijv_minor"],
        "ijv_ecc": feats["ijv_ecc"],
        "ijv_axisratio": feats["ijv_axisratio"],
        "ijv_round": feats["ijv_round"],
    }])

    X = X[columns]
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    X_scaled = scaler.transform(X)

    prob = float(xgb.predict_proba(X_scaled)[0, 1])

    return prob, feats