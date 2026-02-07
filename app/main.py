import time
import cv2

from runtime.hand_tracker import MediaPipeHandTracker
from runtime.features import build_feature_vector
from runtime.intent_engine import IntentEngine, IntentConfig
from runtime.inference import StaticModel


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (try index 1).")

    tracker = MediaPipeHandTracker()
    engine = IntentEngine(IntentConfig())

    # Load trained model
    # model = StaticModel(model_path="models/static_lr.joblib",
    #                     labelmap_path="models/label_map.json")
    model = StaticModel()

    # Confidence threshold to suppress uncertain predictions
    CONF_ACCEPT = 0.70

    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        # Hand landmarks
        hf, _ = tracker.process(frame)
        hands_present = (hf.mask_left == 1 or hf.mask_right == 1)

        # Predict
        if hands_present:
            vec = build_feature_vector(hf)
            label, conf = model.predict(vec)

            # Convert low confidence into UNKNOWN
            if conf < CONF_ACCEPT:
                label, conf = "UNKNOWN", conf
        else:
            label, conf = "NO_HANDS", 1.0

        # Touchless commit logic
        ui = engine.update(label, conf, hands_present)

        # FPS
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev))
        prev = now

        # ------------ Overlay UI ------------
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Hands: {'YES' if hands_present else 'NO'}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.putText(frame, f"Live: {ui['live_label']} ({ui['live_conf']:.2f})", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cand = ui["candidate"]
        if cand:
            # cand = ("WORD"/"CMD", label, conf)
            cv2.putText(frame, f"Candidate: {cand[1]} ({cand[2]:.2f})", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Candidate: -", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.putText(frame, f"Sentence: {ui['transcript']}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, "Tip: show sign, then drop hands (rest) to commit", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.putText(frame, f"CONF_ACCEPT={CONF_ACCEPT:.2f} (below => UNKNOWN)", (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("SignToSpeech - Live Demo (Static Model)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
