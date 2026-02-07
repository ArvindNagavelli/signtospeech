# runtime/features.py
import numpy as np
from .hand_tracker import HandFrame

WRIST = 0
MIDDLE_MCP = 9


def _normalize_hand(hand: np.ndarray, present: int) -> np.ndarray:
    """
    Normalize per hand:
    - Translate so wrist is origin
    - Scale by wrist->middle_mcp distance
    """
    if present == 0:
        return hand

    pts = hand.copy()
    origin = pts[WRIST, :2]  # x,y
    pts[:, :2] -= origin

    scale = np.linalg.norm(pts[MIDDLE_MCP, :2])  # distance after translation
    if scale < 1e-6:
        scale = 1.0
    pts[:, :2] /= scale

    return pts


def build_feature_vector(hf: HandFrame) -> np.ndarray:
    """
    Output vector length 128:
      [left(63), right(63), mask_left, mask_right]
    """
    left = _normalize_hand(hf.left, hf.mask_left).reshape(-1)   # 63
    right = _normalize_hand(hf.right, hf.mask_right).reshape(-1) # 63
    vec = np.concatenate([left, right, np.array([hf.mask_left, hf.mask_right], dtype=np.float32)], axis=0)
    return vec.astype(np.float32)
