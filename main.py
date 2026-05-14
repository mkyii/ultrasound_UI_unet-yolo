import os
import cv2
import time
import queue
import joblib
import torch
import threading
import numpy as np

from ultralytics import YOLO
from torchvision import transforms

from CFG import parse_args
from models.loader import load_model

from inference.detector import yolo_only
from inference.worker import background_worker

from visualization.dashboard import draw_boxes, make_status_banner

from utils.io import (
    collect_samples_by_folder,
    remove_existing_files,
    save_features_to_csv,
    save_patient_result,
    get_first_frame_info,
    create_video_writer,
)

from utils.metric import evaluate_patient_level


def save_mask_npy(final_mask, folder, fname, args):
    if not args.save_npy:
        return

    patient_npy_dir = os.path.join(args.npy_dir, folder)
    os.makedirs(patient_npy_dir, exist_ok=True)

    npy_name = os.path.splitext(fname)[0] + ".npy"
    npy_path = os.path.join(patient_npy_dir, npy_name)

    np.save(npy_path, final_mask.astype(np.uint8))


def main():
    args = parse_args()

    remove_existing_files([
        args.frame_eval_csv_path,
        args.patient_eval_csv_path,
    ])

    task_queue = queue.Queue(maxsize=args.task_queue_size)
    result_queue = queue.Queue(maxsize=args.result_queue_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo_model = YOLO(args.yolo_path)

    unet_model = load_model(args.unet_path, device)
    unet_model.eval()

    xgb_model = joblib.load(args.xgb_model_path)
    xgb_scaler = joblib.load(args.xgb_scaler_path)

    crop_transform = transforms.Compose([
        transforms.Resize((args.unet_input_size, args.unet_input_size)),
        transforms.ToTensor(),
    ])

    samples_by_folder = collect_samples_by_folder(args.img_dir)
    total_samples = sum(len(v) for v in samples_by_folder.values())

    print(f"Total samples: {total_samples}")
    print(f"Total folders/patients: {len(samples_by_folder)}")

    _, H, W = get_first_frame_info(samples_by_folder)

    writer, _ = create_video_writer(
        video_out_path=args.video_out_path,
        width=W,
        height=H,
        banner_width=args.banner_width,
        fps=args.video_fps,
    )

    worker = threading.Thread(
        target=background_worker,
        kwargs={
            "task_queue": task_queue,
            "result_queue": result_queue,
            "unet_model": unet_model,
            "crop_transform": crop_transform,
            "xgb_model": xgb_model,
            "xgb_scaler": xgb_scaler,
            "device": device,
            "args": args,
        },
        daemon=True,
    )
    worker.start()

    frame_id = 0
    prev_time = time.time()
    ijv_start_threshold = 1.0

    frame_name_map = {}

    for folder, samples in samples_by_folder.items():
        print(f"\n[RESET] New patient/folder: {folder}")

        pred_history = []
        prob_history = []
        ca_ecc_history = []
        ijv_ecc_history = []
        used_result_ids = set()
        majority_prob = 0.0
        monitoring_started = False

        for fname, img_path in samples:
            frame = cv2.imread(img_path)

            if frame is None:
                print(f"[SKIP] image load fail: {img_path}")
                continue

            frame_id += 1
            frame_name_map[frame_id] = (folder, fname)

            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))

            yolo_t0 = time.time()

            det_info = yolo_only(
                frame_bgr=frame,
                yolo_model=yolo_model,
                args=args,
            )

            yolo_time = time.time() - yolo_t0

            if not task_queue.full():
                task_queue.put((frame_id, frame.copy(), det_info))

            while not result_queue.empty():
                last_result = result_queue.get()
                result_id = last_result["frame_id"]

                if result_id in used_result_ids:
                    continue

                used_result_ids.add(result_id)

                result_folder, result_fname = frame_name_map.get(
                    result_id,
                    (folder, f"frame_{result_id}")
                )

                rosc_prob = float(last_result["rosc_prob"])
                features = last_result["features"]
                final_mask = last_result["mask"]

                no_detection = len(last_result.get("det_info", [])) == 0

                if no_detection:
                    ca_val = 1.0
                    ijv_val = 1.0
                else:
                    ca_val = features.get("art_ecc", 0.0)
                    ijv_val = features.get("ijv_ecc", 0.0)

                    ca_val = ca_val if ca_val > 0 else 1.0
                    ijv_val = ijv_val if ijv_val > 0 else 1.0

                ca_ecc_history.append(ca_val)
                ijv_ecc_history.append(ijv_val)

                ca_ecc_history = ca_ecc_history[-50:]
                ijv_ecc_history = ijv_ecc_history[-50:]

                if not monitoring_started and ijv_val >= ijv_start_threshold:
                    monitoring_started = True
                    print(
                        f"[MONITOR START] {folder} | "
                        f"frame={result_id} | ijv_ecc={ijv_val:.3f}"
                    )

                frame_pred = int(rosc_prob >= 0.5)

                prob_history.append(rosc_prob)
                pred_history.append(frame_pred)

                majority_prob = sum(pred_history) / max(1, len(pred_history))

                save_dict = features.copy()
                save_dict["rosc_prob"] = rosc_prob
                save_dict["majority_prob"] = float(majority_prob)
                save_dict["frame_pred"] = frame_pred
                save_dict["patient"] = result_folder
                save_dict["monitoring_started"] = int(monitoring_started)
                save_dict["warmup"] = int(not monitoring_started)
                save_dict["ijv_ecc_start_threshold"] = ijv_start_threshold

                save_features_to_csv(
                    image_name=f"{result_folder}/{result_fname}",
                    features_dict=save_dict,
                    csv_path=args.frame_eval_csv_path,
                )

                save_mask_npy(
                    final_mask=final_mask,
                    folder=result_folder,
                    fname=result_fname,
                    args=args,
                )

            vis = draw_boxes(frame, det_info)

            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-8)
            prev_time = now

            CAC = ca_ecc_history[-1] if len(ca_ecc_history) > 0 else 1.0

            SBP = -221.08 * CAC + 254.78
            DBP = -106.75 * CAC + 129.02
            MBP = -146.27 * CAC + 172.25

            display_prob = majority_prob if monitoring_started else -1.0

            banner = make_status_banner(
                height=vis.shape[0],
                width=args.banner_width,
                rosc_prob=display_prob,
                ca_history=ca_ecc_history,
                ijv_history=ijv_ecc_history,
                CAC=CAC,
                SBP=SBP,
                DBP=DBP,
                MBP=MBP,
            )

            combined = np.hstack([vis, banner])

            writer.write(combined)
            cv2.imshow("Realtime ROSC Monitor", combined)

            if frame_id % 10 == 0:
                status_text = "MONITORING" if monitoring_started else "WARMUP"

                print(
                    f"[{frame_id}/{total_samples}] "
                    f"{folder}/{fname} | "
                    f"{status_text} | "
                    f"FPS {fps:.2f} | "
                    f"YOLO {yolo_time * 1000:.1f} ms | "
                    f"ROSC votes: {sum(pred_history)}/{len(pred_history)} | "
                    f"majority_prob: {majority_prob:.3f}"
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        task_queue.join()

        while not result_queue.empty():
            last_result = result_queue.get()
            result_id = last_result["frame_id"]

            if result_id in used_result_ids:
                continue

            used_result_ids.add(result_id)

            result_folder, result_fname = frame_name_map.get(
                result_id,
                (folder, f"remaining_result_{result_id}.png")
            )

            rosc_prob = float(last_result["rosc_prob"])
            features = last_result["features"]
            final_mask = last_result["mask"]

            no_detection = len(last_result.get("det_info", [])) == 0

            if no_detection:
                ca_val = 1.0
                ijv_val = 1.0
            else:
                ca_val = features.get("art_ecc", 0.0)
                ijv_val = features.get("ijv_ecc", 0.0)

                ca_val = ca_val if ca_val > 0 else 1.0
                ijv_val = ijv_val if ijv_val > 0 else 1.0

            if not monitoring_started and ijv_val >= ijv_start_threshold:
                monitoring_started = True
                print(
                    f"[MONITOR START] {folder} | "
                    f"frame={result_id} | ijv_ecc={ijv_val:.3f}"
                )

            frame_pred = int(rosc_prob >= 0.5)

            prob_history.append(rosc_prob)
            pred_history.append(frame_pred)

            majority_prob = sum(pred_history) / max(1, len(pred_history))

            save_dict = features.copy()
            save_dict["rosc_prob"] = rosc_prob
            save_dict["majority_prob"] = float(majority_prob)
            save_dict["frame_pred"] = frame_pred
            save_dict["patient"] = result_folder
            save_dict["monitoring_started"] = int(monitoring_started)
            save_dict["warmup"] = int(not monitoring_started)
            save_dict["ijv_ecc_start_threshold"] = ijv_start_threshold

            save_features_to_csv(
                image_name=f"{result_folder}/{result_fname}",
                features_dict=save_dict,
                csv_path=args.frame_eval_csv_path,
            )

            save_mask_npy(
                final_mask=final_mask,
                folder=result_folder,
                fname=result_fname,
                args=args,
            )

        save_patient_result(
            patient_id=folder,
            prob_history=prob_history,
            pred_history=pred_history,
            csv_path=args.patient_eval_csv_path,
        )

    task_queue.put(None)
    task_queue.join()
    worker.join(timeout=1)

    writer.release()
    cv2.destroyAllWindows()

    print(f"Saved video: {args.video_out_path}")
    print(f"Saved frame-level CSV: {args.frame_eval_csv_path}")
    print(f"Saved patient-level CSV: {args.patient_eval_csv_path}")

    if args.save_npy:
        print(f"Saved npy masks: {args.npy_dir}")

    evaluate_patient_level(
        csv_path=args.patient_eval_csv_path,
        save_path=args.patient_eval_csv_path,
        threshold=0.5,
    )


if __name__ == "__main__":
    main()