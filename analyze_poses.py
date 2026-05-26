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

# landmark indices for the 33 body keypoints MediaPipe gives me
# these are the joints I care about for measuring pilates pose angles
IDX = {
    "LEFT_SHOULDER": 11,  "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW":    13,  "RIGHT_ELBOW":    14,
    "LEFT_WRIST":    15,  "RIGHT_WRIST":    16,
    "LEFT_HIP":      23,  "RIGHT_HIP":      24,
    "LEFT_KNEE":     25,  "RIGHT_KNEE":     26,
    "LEFT_ANKLE":    27,  "RIGHT_ANKLE":    28,
}

# skeleton connections so I can draw the bones between joints
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

# each angle is defined by three joints: (name, point A, vertex, point C)
# the angle gets measured at the vertex between the two rays going to A and C
# this is the same dot product formula from HW0 euclidean distance I just applied
# to vectors between joints instead of coordinate arrays
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

# colors for drawing the skeleton
# left side = yellow, right side = blue, middle = green
LEFT_COLOR  = (0, 200, 255)
RIGHT_COLOR = (255, 100, 0)
MID_COLOR   = (180, 255, 180)
DOT_COLOR   = (255, 255, 255)
TEXT_COLOR  = (50, 255, 50)

LEFT_SIDE  = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
RIGHT_SIDE = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}


def load_image_bgr(path):
    """Load an image from disk as a BGR numpy array.

    Args:
        path (str): path to the image file.

    Returns:
        np.ndarray: image array of shape (H, W, 3) in BGR format.
    """
    img_pil = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def angle_between(a, vertex, c):
    """Compute the angle in degrees at a joint vertex given three landmarks.

    Uses the dot product formula: cos(theta) = (v1 . v2) / (|v1| * |v2|)
    This is the same vector math from HW0 but applied to 3D joint positions
    instead of image coordinates.

    Args:
        a (landmark): first landmark (e.g. shoulder).
        vertex (landmark): the joint where the angle is measured (e.g. elbow).
        c (landmark): third landmark (e.g. wrist).

    Returns:
        float: angle in degrees, or None if vectors have zero length.
    """
    v1 = np.array([a.x - vertex.x, a.y - vertex.y, a.z - vertex.z])
    v2 = np.array([c.x - vertex.x, c.y - vertex.y, c.z - vertex.z])

    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None

    cos_angle = np.dot(v1, v2) / (n1 * n2)
    return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))


def extract_angles(landmarks):
    """Extract all joint angles from a set of pose landmarks.

    Skips any angle where one of the three joints has low visibility
    since occluded joints give unreliable angle readings.

    Args:
        landmarks (list): list of 33 MediaPipe pose landmarks.

    Returns:
        dict: maps angle name (str) to angle in degrees (float) or None if occluded.
    """
    angles = {}
    for name, a_key, v_key, c_key in ANGLE_DEFS:
        a      = landmarks[IDX[a_key]]
        vertex = landmarks[IDX[v_key]]
        c      = landmarks[IDX[c_key]]

        # skip if any of the three joints is hidden or occluded
        if min(a.visibility, vertex.visibility, c.visibility) < 0.3:
            angles[name] = None
        else:
            angles[name] = angle_between(a, vertex, c)

    return angles


def connection_color(idx_a, idx_b):
    """Pick color for a skeleton bone based on which side of the body it is on.

    Args:
        idx_a (int): landmark index of first joint.
        idx_b (int): landmark index of second joint.

    Returns:
        tuple: BGR color tuple.
    """
    if idx_a in LEFT_SIDE and idx_b in LEFT_SIDE:
        return LEFT_COLOR
    if idx_a in RIGHT_SIDE and idx_b in RIGHT_SIDE:
        return RIGHT_COLOR
    return MID_COLOR


def draw_skeleton(image, landmarks, angles):
    """Draw skeleton connections, joint dots, and angle labels on an image.

    Args:
        image (np.ndarray): original BGR image.
        landmarks (list): 33 MediaPipe pose landmarks.
        angles (dict): joint name -> angle in degrees from extract_angles().

    Returns:
        np.ndarray: annotated BGR image with skeleton and angle labels drawn on it.
    """
    out = image.copy()
    h, w = out.shape[:2]

    # draw the bone connections between joints
    for a_idx, b_idx in POSE_CONNECTIONS:
        lm_a = landmarks[a_idx]
        lm_b = landmarks[b_idx]
        if lm_a.visibility < 0.3 or lm_b.visibility < 0.3:
            continue
        px_a = (int(lm_a.x * w), int(lm_a.y * h))
        px_b = (int(lm_b.x * w), int(lm_b.y * h))
        cv2.line(out, px_a, px_b, connection_color(a_idx, b_idx), 3, cv2.LINE_AA)

    # draw a dot at each visible joint
    for lm in landmarks:
        if lm.visibility < 0.3:
            continue
        px = (int(lm.x * w), int(lm.y * h))
        cv2.circle(out, px, 5, DOT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(out, px, 5, (0, 0, 0), 1, cv2.LINE_AA)

    # draw the angle label next to each vertex joint
    vertex_map = {name: IDX[v_key] for name, _, v_key, _ in ANGLE_DEFS}
    for name, angle in angles.items():
        if angle is None:
            continue
        v = landmarks[vertex_map[name]]
        if v.visibility < 0.3:
            continue
        px, py = int(v.x * w), int(v.y * h)
        label = f"{name}: {angle:.0f}°"
        # draw shadow first so the text is readable on any background
        cv2.putText(out, label, (px + 6, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, label, (px + 6, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    return out


def make_bar_chart(all_angles, image_names, out_path):
    """Make a grouped bar chart comparing joint angles across all poses.

    Each group of bars corresponds to one joint angle, and each bar within
    the group corresponds to one image. Hatched bars mean the joint was
    occluded and the angle couldn't be measured.

    Args:
        all_angles (list): list of angle dicts, one per image.
        image_names (list): list of image name strings.
        out_path (str): path to save the chart image.
    """
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
                      width=width, label=img_name, color=colors[i],
                      alpha=0.88, edgecolor="white", linewidth=0.5)

        for bar, v in zip(bars, vals):
            if v is None:
                # hatched bar means angle couldn't be measured
                bar.set_hatch("///")
                bar.set_alpha(0.25)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.5,
                        f"{v:.0f}°", ha="center", va="bottom",
                        fontsize=6.5, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(angle_names, fontsize=11)
    ax.set_ylabel("Angle (degrees)", fontsize=12)
    ax.set_title("Joint Angle Comparison — Pilates Poses", fontsize=14,
                 fontweight="bold", pad=14)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 215)
    ax.yaxis.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved → {out_path}")


def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(folder, "pose_landmarker.task")

    if not os.path.exists(model_path):
        sys.exit(f"Model file not found: {model_path}\n"
                 "Download from: https://storage.googleapis.com/mediapipe-models/"
                 "pose_landmarker/pose_landmarker_heavy/float16/latest/"
                 "pose_landmarker_heavy.task")

    # grab all image files from the folder
    exts = {".jpg", ".jpeg", ".webp", ".avif", ".png"}
    image_paths = sorted(
        p for p in (os.path.join(folder, f) for f in os.listdir(folder))
        if os.path.splitext(p)[1].lower() in exts
        and not os.path.basename(p).startswith(".")
    )

    if not image_paths:
        sys.exit("No images found in folder.")

    print(f"Found {len(image_paths)} images")

    out_dir = os.path.join(folder, "output")
    os.makedirs(out_dir, exist_ok=True)

    # set up the MediaPipe pose detector
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )

    all_angles  = []
    image_names = []

    with mp_vision.PoseLandmarker.create_from_options(opts) as detector:
        for path in image_paths:
            base = os.path.splitext(os.path.basename(path))[0]
            print(f"\nProcessing {os.path.basename(path)}")

            # load image and run pose detection
            img_bgr = load_image_bgr(path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = detector.detect(mp_image)

            if not result.pose_landmarks:
                print("  no pose detected")
                annotated = img_bgr.copy()
                cv2.putText(annotated, "No pose detected", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 220), 3)
                angles = {name: None for name, *_ in ANGLE_DEFS}
            else:
                landmarks = result.pose_landmarks[0]
                angles    = extract_angles(landmarks)
                annotated = draw_skeleton(img_bgr, landmarks, angles)

                # print out the angle readings for this image
                for name, val in angles.items():
                    print(f"  {name}: {f'{val:.1f}°' if val is not None else 'n/a'}")

            # save the skeleton overlay
            out_path = os.path.join(out_dir, f"{base}_skeleton.jpg")
            cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  saved → {out_path}")

            all_angles.append(angles)
            image_names.append(base)

    # generate the comparison bar chart across all poses
    print("\nGenerating joint angle comparison chart...")
    chart_path = os.path.join(out_dir, "joint_angle_comparison.png")
    make_bar_chart(all_angles, image_names, chart_path)

    print(f"\nDone — {len(image_paths)} images processed → {out_dir}/")


if __name__ == "__main__":
    main()
