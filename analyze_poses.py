import os
import sys
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Skeleton connections (BlazePose 33-landmark topology) ────────────────────
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# Landmark indices
IDX = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,      "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,     "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,    "RIGHT_ANKLE": 28,
}

# (display_name, point_a_key, vertex_key, point_c_key)
ANGLE_DEFS = [
    ("L Elbow",    "LEFT_SHOULDER",  "LEFT_ELBOW",    "LEFT_WRIST"),
    ("R Elbow",    "RIGHT_SHOULDER", "RIGHT_ELBOW",   "RIGHT_WRIST"),
    ("L Shoulder", "LEFT_ELBOW",     "LEFT_SHOULDER", "LEFT_HIP"),
    ("R Shoulder", "RIGHT_ELBOW",    "RIGHT_SHOULDER","RIGHT_HIP"),
    ("L Hip",      "LEFT_SHOULDER",  "LEFT_HIP",      "LEFT_KNEE"),
    ("R Hip",      "RIGHT_SHOULDER", "RIGHT_HIP",     "RIGHT_KNEE"),
    ("L Knee",     "LEFT_HIP",       "LEFT_KNEE",     "LEFT_ANKLE"),
    ("R Knee",     "RIGHT_HIP",      "RIGHT_KNEE",    "RIGHT_ANKLE"),
]

# ── Colours ──────────────────────────────────────────────────────────────────
LEFT_COLOR  = (0, 200, 255)   # yellow-ish
RIGHT_COLOR = (255, 100, 0)   # blue-ish
MID_COLOR   = (180, 255, 180) # light green
DOT_COLOR   = (255, 255, 255)
TEXT_COLOR  = (50, 255, 50)

LEFT_SIDE   = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
RIGHT_SIDE  = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}


def load_image_bgr(path):
    img_pil = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def angle_between(a, vertex, c):
    """Angle in degrees at `vertex`."""
    v1 = np.array([a.x - vertex.x, a.y - vertex.y, a.z - vertex.z])
    v2 = np.array([c.x - vertex.x, c.y - vertex.y, c.z - vertex.z])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def extract_angles(landmarks):
    angles = {}
    for name, a_key, v_key, c_key in ANGLE_DEFS:
        a = landmarks[IDX[a_key]]
        v = landmarks[IDX[v_key]]
        c = landmarks[IDX[c_key]]
        if min(a.visibility, v.visibility, c.visibility) < 0.3:
            angles[name] = None
        else:
            angles[name] = angle_between(a, v, c)
    return angles


def connection_color(idx_a, idx_b):
    if idx_a in LEFT_SIDE and idx_b in LEFT_SIDE:
        return LEFT_COLOR
    if idx_a in RIGHT_SIDE and idx_b in RIGHT_SIDE:
        return RIGHT_COLOR
    return MID_COLOR


def draw_skeleton(image, landmarks, angles):
    out = image.copy()
    h, w = out.shape[:2]

    # Draw connections
    for a_idx, b_idx in POSE_CONNECTIONS:
        lm_a = landmarks[a_idx]
        lm_b = landmarks[b_idx]
        if lm_a.visibility < 0.3 or lm_b.visibility < 0.3:
            continue
        px_a = (int(lm_a.x * w), int(lm_a.y * h))
        px_b = (int(lm_b.x * w), int(lm_b.y * h))
        color = connection_color(a_idx, b_idx)
        cv2.line(out, px_a, px_b, color, 3, cv2.LINE_AA)

    # Draw landmark dots
    for lm in landmarks:
        if lm.visibility < 0.3:
            continue
        px = (int(lm.x * w), int(lm.y * h))
        cv2.circle(out, px, 5, DOT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(out, px, 5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw angle labels at vertices
    vertex_map = {name: IDX[v_key] for name, _, v_key, _ in ANGLE_DEFS}
    for name, angle in angles.items():
        if angle is None:
            continue
        v = landmarks[vertex_map[name]]
        if v.visibility < 0.3:
            continue
        px, py = int(v.x * w), int(v.y * h)
        label = f"{name}: {angle:.0f}°"
        # Shadow
        cv2.putText(out, label, (px + 6, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, label, (px + 6, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    return out


def make_bar_chart(all_angles, image_names, out_path):
    angle_names = [name for name, *_ in ANGLE_DEFS]
    n_images = len(image_names)
    n_angles = len(angle_names)

    x = np.arange(n_angles)
    width = 0.8 / n_images
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_images))

    fig, ax = plt.subplots(figsize=(max(14, n_angles * 1.8), 7))

    for i, (img_name, angles) in enumerate(zip(image_names, all_angles)):
        vals = [angles.get(a) for a in angle_names]
        offsets = x + (i - n_images / 2 + 0.5) * width
        bars = ax.bar(offsets, [v if v is not None else 0 for v in vals],
                      width=width, label=img_name, color=colors[i], alpha=0.88,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v is None:
                bar.set_hatch("///")
                bar.set_alpha(0.25)
            else:
                # Value label on bar
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.5,
                        f"{v:.0f}°", ha="center", va="bottom",
                        fontsize=6.5, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(angle_names, fontsize=11)
    ax.set_ylabel("Angle (degrees)", fontsize=12)
    ax.set_title("Joint Angle Comparison — Pilates Poses", fontsize=14, fontweight="bold", pad=14)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 215)
    ax.yaxis.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(folder, "pose_landmarker.task")
    if not os.path.exists(model_path):
        sys.exit(f"Model file not found: {model_path}\n"
                 "Download from: https://storage.googleapis.com/mediapipe-models/"
                 "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")

    exts = {".jpg", ".jpeg", ".webp", ".avif", ".png"}
    image_paths = sorted(
        p for p in (os.path.join(folder, f) for f in os.listdir(folder))
        if os.path.splitext(p)[1].lower() in exts
           and not os.path.basename(p).startswith(".")
    )
    if not image_paths:
        sys.exit("No images found.")

    out_dir = os.path.join(folder, "output")
    os.makedirs(out_dir, exist_ok=True)

    # Build detector
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )

    all_angles = []
    image_names = []

    with mp_vision.PoseLandmarker.create_from_options(opts) as detector:
        for path in image_paths:
            base = os.path.splitext(os.path.basename(path))[0]
            print(f"Processing {os.path.basename(path)} …")

            img_bgr = load_image_bgr(path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = detector.detect(mp_image)

            if not result.pose_landmarks:
                print("  ! No pose detected")
                annotated = img_bgr.copy()
                cv2.putText(annotated, "No pose detected", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 220), 3)
                angles = {name: None for name, *_ in ANGLE_DEFS}
            else:
                landmarks = result.pose_landmarks[0]
                angles = extract_angles(landmarks)
                annotated = draw_skeleton(img_bgr, landmarks, angles)

                # Print angle summary
                for name, val in angles.items():
                    flag = f"{val:.1f}°" if val is not None else "n/a"
                    print(f"    {name}: {flag}")

            out_path = os.path.join(out_dir, f"{base}_skeleton.jpg")
            cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  saved → {out_path}")

            all_angles.append(angles)
            image_names.append(base)

    chart_path = os.path.join(out_dir, "joint_angle_comparison.png")
    print("\nGenerating comparison bar chart …")
    make_bar_chart(all_angles, image_names, chart_path)

    print(f"\nDone — {len(image_paths)} images → {out_dir}/")


if __name__ == "__main__":
    main()
