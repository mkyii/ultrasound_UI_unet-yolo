import time
import queue

from classifier import compute_rosc_probability
from inference.segmenter import unet_from_yolo_detections
from utils.features import get_default_features, features_to_array
from visualization.dashboard import make_overlay


def background_worker(
    task_queue,
    result_queue,
    unet_model,
    crop_transform,
    xgb_model,
    xgb_scaler,
    device,
    args,
):
    """
    main loop와 분리된 thread에서 segmentation과 ROSC classification을 처리한다.
    """
    while True:
        item = task_queue.get()

        if item is None:
            task_queue.task_done()
            break

        frame_id, frame_bgr, det_info = item

        t0 = time.time()

        final_mask = unet_from_yolo_detections(
            frame_bgr=frame_bgr,
            det_info=det_info,
            unet_model=unet_model,
            crop_transform=crop_transform,
            device=device,
            args=args,
        )

        if len(det_info) == 0:
            B2 = get_default_features()

            X = features_to_array(B2)
            X = xgb_scaler.transform(X)

            rosc_prob = float(xgb_model.predict_proba(X)[0, 1])

        else:
            rosc_prob, B2 = compute_rosc_probability(
                final_mask,
                xgb_model,
                xgb_scaler
            )

        overlay = make_overlay(frame_bgr, final_mask)

        elapsed = time.time() - t0

        result = {
            "frame_id": frame_id,
            "mask": final_mask,
            "overlay": overlay,
            "rosc_prob": rosc_prob,
            "features": B2,
            "worker_time": elapsed,
            "det_info": det_info,
        }

        if result_queue.full():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass

        result_queue.put(result)
        task_queue.task_done()