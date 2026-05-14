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
    H, W = frame_bgr.shape[:2]
    final_mask = np.zeros((H, W), dtype=np.uint8)

    if len(det_info) == 0:
        return final_mask

    crop_tensors = []
    crop_meta = []

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

        crop_tensors.append(crop_tensor)

        crop_meta.append({
            "class": cls,
            "box": [x1, y1, x2, y2],
            "crop_size": (crop_h, crop_w)
        })

    if len(crop_tensors) == 0:
        return final_mask

    batch_tensor = torch.cat(crop_tensors, dim=0)

    pred = unet_model(batch_tensor)
    prob = torch.sigmoid(pred)
    pred_masks = (prob > args.unet_threshold).float()
    pred_masks = pred_masks[:, 0].cpu().numpy().astype(np.uint8)

    for pred_mask, meta in zip(pred_masks, crop_meta):
        cls = meta["class"]
        x1, y1, x2, y2 = meta["box"]
        crop_h, crop_w = meta["crop_size"]

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