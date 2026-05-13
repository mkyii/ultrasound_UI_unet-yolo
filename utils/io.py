import os
import csv
import cv2
import numpy as np


def collect_samples_by_folder(img_dir):
    """
    폴더 하나를 patient 하나로 간주해 frame 목록을 구성한다.
    """
    samples_by_folder = {}

    for folder in sorted(os.listdir(img_dir)):
        img_folder = os.path.join(img_dir, folder)

        if not os.path.isdir(img_folder):
            continue

        samples_by_folder[folder] = []

        for fname in sorted(os.listdir(img_folder)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue

            img_path = os.path.join(img_folder, fname)
            samples_by_folder[folder].append((fname, img_path))

    if len(samples_by_folder) == 0:
        raise FileNotFoundError("처리할 이미지가 없습니다.")

    return samples_by_folder


def remove_existing_files(paths):
    """
    이전 실행 결과가 누적되지 않도록 기존 결과 파일을 삭제한다.
    """
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def save_features_to_csv(image_name, features_dict, csv_path):
    """
    Frame별 feature와 prediction 결과를 CSV에 누적 저장한다.
    """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image"] + list(features_dict.keys())
        )

        if not file_exists:
            writer.writeheader()

        row = {"image": image_name}
        row.update(features_dict)

        writer.writerow(row)


def save_patient_result(patient_id, prob_history, pred_history, csv_path):
    """
    Frame별 ROSC probability를 patient 단위로 평균내어 최종 patient prediction을 저장한다.
    """
    if len(prob_history) == 0:
        return

    patient_prob = float(np.mean(prob_history))
    patient_pred = int(patient_prob >= 0.5)

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient",
                "prob",
                "pred",
                "num_frames",
                "num_rosc_votes",
                "num_arrest_votes"
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "patient": patient_id,
            "prob": patient_prob,
            "pred": patient_pred,
            "num_frames": len(prob_history),
            "num_rosc_votes": int(sum(pred_history)),
            "num_arrest_votes": int(len(pred_history) - sum(pred_history))
        })

    print(
        f"[PATIENT SAVE] {patient_id} | "
        f"prob={patient_prob:.3f} | pred={patient_pred} | frames={len(prob_history)}"
    )


def get_first_frame_info(samples_by_folder):
    """
    첫 frame의 해상도를 기준으로 전체 output video 크기를 결정한다.
    """
    first_folder = list(samples_by_folder.keys())[0]
    first_frame_path = samples_by_folder[first_folder][0][1]

    first_frame = cv2.imread(first_frame_path)

    if first_frame is None:
        raise FileNotFoundError(f"첫 이미지 로드 실패: {first_frame_path}")

    height, width = first_frame.shape[:2]

    return first_frame, height, width


def create_video_writer(video_out_path, width, height, banner_width, fps):
    """
    원본 frame과 side banner를 합친 크기로 VideoWriter를 생성한다.
    """
    out_size = (width + banner_width, height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer = cv2.VideoWriter(
        video_out_path,
        fourcc,
        fps,
        out_size
    )

    return writer, out_size