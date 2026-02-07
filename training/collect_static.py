import os
import time
import json
import cv2
import numpy as np
import mediapipe as mp

from runtime.hand_tracker import MediaPipeHandTracker
from runtime.features import build_feature_vector

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# ---------------- CONFIG ----------------
LABELS = ["HELLO", "THANK_YOU", "YES", "CONFIRM"]

SAMPLES_PER_LABEL = 60          # target samples per sign
HOLD_FRAMES = 5                 # frames sign must be stable
STABILITY_DIFF_THRESHOLD = 0.06 # increase to 0.08 if still not capturing
SAVE_ROOT = "data/static"
# ----------------------------------------


def ensure_dirs():
    for label in LABELS:
        os.makedirs(os.path.join(SAVE_ROOT, label), exist_ok=True)
def save_sample(vec, label):
    ts = int(time.time() * 1000)
    path = os.path.join(SAVE_ROOT, label, f"{label}_{ts}.json")
    with open(path, "w") as f:
        json.dump(vec.tolist(), f)
    return path


def main():
    ensure_dirs()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (try index 1)")

    tracker = MediaPipeHandTracker()

    label_index = 0
    current_label = LABELS[label_index]
    sample_count = 0
    hold_counter = 0

    prev_vec = None
    last_diff = None

    print("Instructions:")
    print("  - Show the sign clearly")
    print("  - Hold steady until it is captured")
    print("  - Press N for next sign, P for previous")
    print("  - ESC to exit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        # IMPORTANT: capture 'res' so we can draw landmarks
        hf, res = tracker.process(frame)

        # Draw detected landmarks so we know hand detection works
        if res and res.multi_hand_landmarks:
            for hlm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        hands_present = (hf.mask_left == 1 or hf.mask_right == 1)

        vec = None
        if hands_present:
            vec = build_feature_vector(hf)

            if prev_vec is not None:
                diff = float(np.linalg.norm(vec - prev_vec))
                last_diff = diff

                if diff < STABILITY_DIFF_THRESHOLD:
                    hold_counter += 1
                else:
                    hold_counter = 0
            else:
                hold_counter = 0

            prev_vec = vec
        else:
            hold_counter = 0
            prev_vec = None
            last_diff = None

        # Capture sample only when stable
        if (hands_present and vec is not None and
                hold_counter >= HOLD_FRAMES and sample_count < SAMPLES_PER_LABEL):

            ts = int(time.time() * 1000)
            path = os.path.join(SAVE_ROOT, current_label, f"{current_label}_{ts}.json")

            with open(path, "w") as f:
                json.dump(vec.tolist(), f)

            sample_count += 1
            hold_counter = 0
            print(f"[{current_label}] captured {sample_count}/{SAMPLES_PER_LABEL}")

        # UI overlay
        cv2.putText(frame, f"LABEL: {current_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, f"SAMPLES: {sample_count}/{SAMPLES_PER_LABEL}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(frame, "Hold sign steady to capture", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Manual + navigation hints
        cv2.putText(frame, "SPACE = manual capture (moving signs)", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, "N = next | P = previous | ESC = exit", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Debug info (Step 3)
        cv2.putText(frame, f"Hands: {'YES' if hands_present else 'NO'}", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"Hold: {hold_counter}/{HOLD_FRAMES}", (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if last_diff is not None:
            cv2.putText(frame, f"Diff: {last_diff:.3f} (thr<{STABILITY_DIFF_THRESHOLD})", (10, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("BSL Static Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        # SPACE = manual capture
        if key == 32:
            if hands_present and vec is not None and sample_count < SAMPLES_PER_LABEL:
                save_sample(vec, current_label)
                sample_count += 1
                hold_counter = 0
                prev_vec = None
                last_diff = None
                print(f"[{current_label}] manual capture {sample_count}/{SAMPLES_PER_LABEL}")

        # ESC = exit
        elif key == 27:
            break

        # N = next label
        elif key == ord('n'):
            label_index = (label_index + 1) % len(LABELS)
            current_label = LABELS[label_index]
            sample_count = 0
            hold_counter = 0
            prev_vec = None
            last_diff = None

        # P = previous label
        elif key == ord('p'):
            label_index = (label_index - 1) % len(LABELS)
            current_label = LABELS[label_index]
            sample_count = 0
            hold_counter = 0
            prev_vec = None
            last_diff = None

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
