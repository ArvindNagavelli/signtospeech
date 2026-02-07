# runtime/hand_tracker.py
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandFrame:
    left: np.ndarray       # shape (21, 3)
    right: np.ndarray      # shape (21, 3)
    mask_left: int         # 1 if present else 0
    mask_right: int        # 1 if present else 0


class MediaPipeHandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame_bgr: np.ndarray) -> Tuple[HandFrame, Optional[object]]:
        """
        Returns a HandFrame (always fixed shape) + raw mediapipe result (for drawing/debug).
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        zero = np.zeros((21, 3), dtype=np.float32)
        left = zero.copy()
        right = zero.copy()
        mask_left = 0
        mask_right = 0

        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handed.classification[0].label  # "Left" or "Right"
                pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)

                # Keep consistent order: left then right
                if label == "Left":
                    left = pts
                    mask_left = 1
                elif label == "Right":
                    right = pts
                    mask_right = 1

        return HandFrame(left=left, right=right, mask_left=mask_left, mask_right=mask_right), res

    def close(self):
        self.hands.close()
