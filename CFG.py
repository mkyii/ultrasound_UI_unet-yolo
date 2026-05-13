import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--class_artery",
        type=int,
        default=0,
        help="YOLO class id for carotid artery"
    )

    parser.add_argument(
        "--class_vein",
        type=int,
        default=1,
        help="YOLO class id for internal jugular vein"
    )

    ##################### Input / output paths #####################
    parser.add_argument(
        "--img_dir",
        type=str,
        default="../Dataset/GE/image",
        help="Input image directory"
    )

    parser.add_argument(
        "--video_out_path",
        type=str,
        default="./GE_time_monitor_all_images.avi",
        help="Output visualization video path"
    )

    parser.add_argument(
        "--frame_eval_csv_path",
        type=str,
        default="./frame_level_eval.csv",
        help="Path to save Frame-level prediction results"
    )

    parser.add_argument(
        "--patient_eval_csv_path",
        type=str,
        default="./patient_level_eval.csv",
        help="Path to save patient-level evaluation metrics"
    )

    ##################### Model checkpoint #####################
    parser.add_argument(
        "--yolo_path",
        type=str,
        default="./checkpoint/emr_yolo.pt",
        help="YOLO checkpoint path"
    )

    parser.add_argument(
        "--unet_path",
        type=str,
        default="./checkpoint/best_unet_20260503.bin",
        help="UNet checkpoint path"
    )

    parser.add_argument(
        "--xgb_model_path",
        type=str,
        default="./checkpoint/xgb_models/xgb_model_fold1.pkl",
        help="XGBoost model path"
    )

    parser.add_argument(
        "--xgb_scaler_path",
        type=str,
        default="./checkpoint/xgb_models/xgb_scaler_fold1.pkl",
        help="Feature scaler path"
    )

    ##################### YOLO inference config #####################

    # YOLO inference config
    parser.add_argument(
        "--yolo_imgsz",
        type=int,
        default=512,
        help="YOLO inference image size"
    )

    parser.add_argument(
        "--yolo_conf",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold"
    )

    ##################### Segmentation postprocess config #####################
    parser.add_argument(
        "--unet_input_size",
        type=int,
        default=256,
        help="UNet input resolution"
    )

    parser.add_argument(
        "--unet_threshold",
        type=float,
        default=0.55,
        help="Segmentation probability threshold"
    )

    ##################### Runtime / visualization config #####################
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Extra bbox padding for vessel crop"
    )

    parser.add_argument(
        "--min_mask_area",
        type=int,
        default=50,
        help="Minimum connected-component area to keep"
    )

    parser.add_argument(
        "--video_fps",
        type=int,
        default=30,
        help="Output video FPS"
    )

    parser.add_argument(
        "--banner_width",
        type=int,
        default=360,
        help="Right-side monitoring panel width"
    )

    parser.add_argument(
        "--task_queue_size",
        type=int,
        default=2,
        help="Max size of inference task queue"
    )

    parser.add_argument(
        "--result_queue_size",
        type=int,
        default=2,
        help="Max size of inference result queue"
    )

    return parser.parse_args()