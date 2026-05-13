import cv2
import numpy as np

def keep_largest_component(binary_mask, min_area=30):
    """
    가장 큰 connected component만 vessel 후보로 유지.
    """
    binary_mask = binary_mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    if num_labels <= 1:
        return binary_mask

    areas = stats[1:, cv2.CC_STAT_AREA]

    if len(areas) == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    largest_idx = 1 + np.argmax(areas)
    largest_area = stats[largest_idx, cv2.CC_STAT_AREA]

    if largest_area < min_area:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    clean_mask = (labels == largest_idx).astype(np.uint8)
    return clean_mask

def fill_holes(binary_mask):
    """
    Vessel 내부가 thresholding 과정에서 비는 경우를 보정하기 위해 hole을 채움.
    """
    binary_mask = binary_mask.astype(np.uint8)

    h, w = binary_mask.shape
    flood = binary_mask.copy()

    mask = np.zeros((h + 2, w + 2), np.uint8)

    # 영상 외곽과 연결된 배경을 먼저 채워 내부 hole과 분리한다.
    cv2.floodFill(flood, mask, (0, 0), 1)

    # flood fill 결과를 반전하면 object 내부 hole만 남는다.
    flood_inv = 1 - flood

    # 원본 mask와 hole 영역을 합쳐 닫힌 vessel mask로 만든다.
    filled = binary_mask | flood_inv

    return filled.astype(np.uint8)

def remove_outside_center_region(binary_mask, keep_ratio=0.85):
    """
    crop 가장자리의 false positive를 줄이기 위해 중심부 영역만 유지.
    """
    H, W = binary_mask.shape

    cx, cy = W // 2, H // 2
    new_w = int(W * keep_ratio)
    new_h = int(H * keep_ratio)

    x1 = max(0, cx - new_w // 2)
    x2 = min(W, cx + new_w // 2)
    y1 = max(0, cy - new_h // 2)
    y2 = min(H, cy + new_h // 2)

    center_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    center_mask[y1:y2, x1:x2] = 1

    return binary_mask * center_mask

def postprocess_pred_mask(pred_mask, min_area=30):
    """
    # UNet binary output에 morphology, hole filling, largest component filtering을 순차 적용.
    """
    pred_mask = pred_mask.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

    pred_mask = fill_holes(pred_mask)

    pred_mask = remove_outside_center_region(pred_mask, keep_ratio=0.90)

    pred_mask = keep_largest_component(pred_mask, min_area=min_area)

    pred_mask = fill_holes(pred_mask)

    return pred_mask
