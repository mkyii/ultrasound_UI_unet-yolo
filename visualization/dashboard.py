import cv2
import numpy as np

def make_overlay(frame_bgr, final_mask, alpha=0.35):
    '''
    artery와 vein segmentation 결과를 원본 frame 위에 반투명하게 표시.
    '''
    overlay = frame_bgr.copy()

    overlay[final_mask == 1] = (0, 0, 255)      # artery red
    overlay[final_mask == 2] = (255, 0, 0)      # vein blue

    return cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)

def draw_boxes(frame_bgr, det_info):
    '''
    YOLO 검출 bbox와 confidence를 frame 위에 표시.
    '''
    vis = frame_bgr.copy()

    for det in det_info:
        x1, y1, x2, y2 = det["box"]
        name = det["name"]
        conf = det["conf"]

        color = (0, 0, 255) if name == "artery" else (255, 0, 0)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            vis,
            f"{name} {conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return vis


def draw_ecc_graph(banner, ca_history, ijv_history, x=30, y=320, w=300, h=160):
    '''
    최근 50개 frame의 CA/IJV eccentricity 변화를 UI 패널 그림.
    '''
    cv2.rectangle(banner, (x, y), (x + w, y + h), (45, 45, 45), -1)
    cv2.rectangle(banner, (x, y), (x + w, y + h), (100, 100, 100), 1)

    def plot_line(values, color):
        if len(values) < 2:
            return

        values = np.array(values[-50:], dtype=np.float32)
        values = np.clip(values, 0, 1)

        pts = []
        for i, v in enumerate(values):
            px = x + int(i / max(1, len(values) - 1) * w)
            py = y + h - int(v * h)
            pts.append((px, py))

        for i in range(1, len(pts)):
            cv2.line(banner, pts[i - 1], pts[i], color, 2)

    plot_line(ca_history, (0, 0, 255))
    plot_line(ijv_history, (255, 80, 40))

    # CA와 IJV line color를 UI에서 구분하기 위한 legend를 표시한다.
    cv2.circle(banner, (x + 15, y + h - 15), 5, (0, 0, 255), -1)
    cv2.putText(banner, "CA", (x + 25, y + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    cv2.circle(banner, (x + 75, y + h - 15), 5, (255, 80, 40), -1)
    cv2.putText(banner, "IJV", (x + 85, y + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    return banner


def make_status_banner(height, width, rosc_prob, ca_history, ijv_history,
                       CAC, SBP, DBP, MBP):
    '''
    ROSC 상태, CAC, 추정 혈압 지표, temporal graph를 하나의 side banner로 구성.
    '''
    banner = np.zeros((height, width, 3), dtype=np.uint8)
    banner[:] = (18, 18, 18)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # patient-level 누적 majority probability를 기준으로 현재 상태를 표시
    if rosc_prob < 0:
        status = "WARM UP"
        status_color = (160, 160, 160)

    elif rosc_prob >= 0.5:
        status = "ROSC"
        status_color = (40, 220, 120)

    else:
        status = "ARREST"
        status_color = (40, 40, 230)

    cv2.rectangle(banner, (15, 15), (width - 15, 55), (35, 35, 35), -1)
    cv2.putText(
        banner,
        "ROSC Status",
        (25, 40),
        font,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )
    cv2.rectangle(banner, (15, 65), (width - 15, 145), (28, 28, 28), -1)

    cv2.putText(banner, f"CAC: {CAC:.3f}",
                (25, 95), font, 0.7, (220, 220, 220), 2)

    cv2.putText(banner, f"SBP: {SBP:.1f}",
                (25, 120), font, 0.6, (0, 200, 255), 2)

    cv2.putText(banner, f"DBP: {DBP:.1f}",
                (150, 120), font, 0.6, (255, 200, 0), 2)

    cv2.putText(banner, f"MBP: {MBP:.1f}",
                (25, 140), font, 0.6, (200, 255, 100), 2)

    cv2.putText(
        banner,
        status,
        (width - 135, 42),
        font,
        0.6,
        status_color,
        2,
        cv2.LINE_AA
    )

    # CAC graph 영역은 현재 frame 기준 최근 eccentricity history를 표시
    cac_x, cac_y = 15, 160
    cac_w, cac_h = width - 30, int((height - 120) * 0.5)

    cv2.rectangle(banner, (cac_x, cac_y), (cac_x + cac_w, cac_y + cac_h), (5, 5, 5), -1)
    cv2.rectangle(banner, (cac_x, cac_y), (cac_x + cac_w, cac_y + cac_h), (45, 45, 45), 1)

    cv2.putText(
        banner,
        "CAC Graph",
        (cac_x + 25, cac_y + 55),
        font,
        0.85,
        (210, 210, 210),
        2,
        cv2.LINE_AA
    )

    draw_ecc_graph(
        banner,
        ca_history,
        ijv_history,
        x=cac_x + 15,
        y=cac_y + 85,
        w=cac_w - 30,
        h=cac_h - 105
    )

    # ABP graph 영역은 추후 실제 ABP waveform
    abp_x, abp_y = 15, cac_y + cac_h + 20
    abp_w, abp_h = width - 30, height - abp_y - 20

    cv2.rectangle(banner, (abp_x, abp_y), (abp_x + abp_w, abp_y + abp_h), (5, 5, 5), -1)
    cv2.rectangle(banner, (abp_x, abp_y), (abp_x + abp_w, abp_y + abp_h), (45, 45, 45), 1)

    cv2.putText(
        banner,
        "ABP Graph",
        (abp_x + 25, abp_y + 70),
        font,
        0.85,
        (210, 210, 210),
        2,
        cv2.LINE_AA
    )

    return banner
