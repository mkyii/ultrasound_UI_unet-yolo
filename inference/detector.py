def add_padding_box(x1, y1, x2, y2, W, H, padding):
    """
    YOLO bbox에 padding을 추가하되 영상 경계를 벗어나지 않도록 보정.
    """
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(W, int(x2) + padding)
    y2 = min(H, int(y2) + padding)

    return x1, y1, x2, y2


def yolo_only(frame_bgr, yolo_model, args):
    """
    실시간 UI thread에서는 YOLO detection만 먼저 수행하고
    segmentation은 worker에 위임.

    동일 class detection이 여러 개 존재할 경우
    confidence가 가장 높은 bbox 하나만 유지.
    """
    results = yolo_model(
        frame_bgr,
        imgsz=args.yolo_imgsz,
        conf=args.yolo_conf,
        verbose=False
    )[0]

    det_info = []

    if results.boxes is None or len(results.boxes) == 0:
        return det_info

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()

    H, W = frame_bgr.shape[:2]

    # class별 최고 confidence detection 저장
    best_det = {}

    for box, cls, conf in zip(boxes, classes, confs):

        if cls not in [args.class_artery, args.class_vein]:
            continue

        # 현재 class보다 더 높은 confidence detection만 유지
        if cls not in best_det or conf > best_det[cls]["conf"]:

            x1, y1, x2, y2 = box

            x1, y1, x2, y2 = add_padding_box(
                x1, y1, x2, y2,
                W, H,
                args.padding
            )

            name = (
                "artery"
                if cls == args.class_artery
                else "vein"
            )

            best_det[cls] = {
                "class": int(cls),
                "name": name,
                "conf": float(conf),
                "box": [x1, y1, x2, y2]
            }

    # artery 1개 + vein 1개만 최종 유지
    det_info = list(best_det.values())

    return det_info