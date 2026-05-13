import cv2
import torch
import numpy as np

from PIL import Image
from inference.postprocess import postprocess_pred_mask


def preprocess_crop(crop_bgr, crop_transform, device):
    """
    OpenCV는 BGR, PIL/torchvision transform은 RGB 기준이므로 channel 순서를 맞춤.
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)
    crop_tensor = crop_transform(crop_pil).unsqueeze(0)

    return crop_tensor.to(device, dtype=torch.float32)

@torch.no_grad()
def unet_from_yolo_detections(
    frame_bgr,
    det_info,
    unet_model,
    crop_transform,
    device,
    args
):
    """  
    YOLO 검출 결과를 입력으로 받아 각 bbox crop에 대해 UNet segmentation을 수행.
    """ 
    H, W = frame_bgr.shape[:2]
    final_mask = np.zeros((H, W), dtype=np.uint8)

    for det in det_info:
        cls = det["class"]
        x1, y1, x2, y2 = det["box"]

        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]

        crop_tensor = preprocess_crop(
            crop_bgr=crop,
            crop_transform=crop_transform,
            device=device
        )

        pred = unet_model(crop_tensor)
        prob = torch.sigmoid(pred)
        pred_mask = (prob > args.unet_threshold).float()
        pred_mask = pred_mask[0, 0].cpu().numpy().astype(np.uint8)

        # crop 기준 mask를 원본 bbox 크기로 복원한다. label mask이므로 nearest interpolation을 사용
        pred_restore = cv2.resize(
            pred_mask,
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST
        )

        pred_restore = postprocess_pred_mask(
            pred_restore,
            min_area=args.min_mask_area
        )

        if pred_restore.sum() == 0:
            continue

        # 최종 mask는 artery=1, vein=2로 저장하여 feature 추출과 시각화에서 동일하게 사용
        if cls == args.class_artery:
            final_mask[y1:y2, x1:x2][pred_restore == 1] = 1
        elif cls == args.class_vein:
            final_mask[y1:y2, x1:x2][pred_restore == 1] = 2

    return final_mask


@torch.no_grad()
def infer_one_frame(
    frame_bgr,
    yolo_model,
    unet_model,
    crop_transform,
    device,
    args
):
    """ 
    단일 프레임에서 YOLO detection, UNet segmentation, class별 mask 병합을 수행.
    """ 
    from inference.detector import yolo_only

    det_info = yolo_only(frame_bgr, yolo_model, args)

    final_mask = unet_from_yolo_detections(
        frame_bgr=frame_bgr,
        det_info=det_info,
        unet_model=unet_model,
        crop_transform=crop_transform,
        device=device,
        args=args
    )

    return final_mask, det_info