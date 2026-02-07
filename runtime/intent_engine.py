# runtime/intent_engine.py
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple


# Commands are special tokens that control the UI flow.
# CONFIRM = end of message / submit current draft to history
COMMANDS = {"DELETE", "CLEAR", "SPEAK", "CONFIRM"}


@dataclass
class IntentConfig:
    window: int = 16
    conf_word: float = 0.85
    conf_cmd: float = 0.90
    stable_frames_word: int = 8
    stable_frames_cmd: int = 10
    rest_frames: int = 6
    cooldown_frames: int = 10


class IntentEngine:
    def __init__(self, cfg: IntentConfig):
        self.cfg = cfg
        self.pred_window: Deque[Tuple[str, float]] = deque(maxlen=cfg.window)

        self.transcript: List[str] = []
        self.state = "IDLE"
        self.candidate: Optional[Tuple[str, str, float]] = None  # ("WORD"/"CMD", label, conf)

        self.rest_counter = 0
        self.stable_counter = 0
        self.last_major_label: Optional[str] = None
        self.cooldown = 0

    def _majority_vote(self) -> Tuple[str, float]:
        counts = {}
        confs = {}
        for lab, c in self.pred_window:
            counts[lab] = counts.get(lab, 0) + 1
            confs[lab] = confs.get(lab, 0.0) + c
        major = max(counts.items(), key=lambda x: x[1])[0]
        avg_conf = confs[major] / counts[major]
        return major, avg_conf

    def update(self, label: str, conf: float, hands_present: bool) -> dict:
        """
        Call once per frame with model output.
        Returns a dict for UI: live_label, live_conf, candidate, transcript, committed
        committed is a single event token when something is committed (word or command).
        """
        # # Rest detection
        # if not hands_present:
        #     self.rest_counter += 1
        # else:
        #     self.rest_counter = 0

        # Rest detection (more robust):
        # Treat "NO_HANDS" as rest even if hands_present is noisy.
        is_rest = (not hands_present) or (label == "NO_HANDS")
        if is_rest:
            self.rest_counter += 1
        else:
            self.rest_counter = 0

        # Update window
        self.pred_window.append((label, conf))
        major_label, avg_conf = self._majority_vote()

        # Stable counter
        if major_label == self.last_major_label:
            self.stable_counter += 1
        else:
            self.stable_counter = 1
            self.last_major_label = major_label

        # Cooldown: do not change candidates / commit during cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            return {
                "live_label": major_label,
                "live_conf": avg_conf,
                "candidate": self.candidate,
                "transcript": " ".join(self.transcript),
                "committed": None
            }

        is_cmd = major_label in COMMANDS
        is_word = (major_label not in {"NO_HANDS", "UNKNOWN"}) and (not is_cmd)

        # Acquire candidate (must be stable + confident)
        if self.state in {"IDLE", "CANDIDATE"}:
            if is_word and self.stable_counter >= self.cfg.stable_frames_word and avg_conf >= self.cfg.conf_word:
                self.candidate = ("WORD", major_label, avg_conf)
                self.state = "CANDIDATE"
            elif is_cmd and self.stable_counter >= self.cfg.stable_frames_cmd and avg_conf >= self.cfg.conf_cmd:
                self.candidate = ("CMD", major_label, avg_conf)
                self.state = "CANDIDATE"

        committed = None

        # Commit on rest (hands lowered)
        if self.state == "CANDIDATE" and self.rest_counter >= self.cfg.rest_frames and self.candidate is not None:
            kind, lab, _ = self.candidate

            if kind == "WORD":
                self.transcript.append(lab)
                committed = lab

            else:
                # Commands
                if lab == "DELETE":
                    if self.transcript:
                        self.transcript.pop()
                    committed = "DELETE"

                elif lab == "CLEAR":
                    self.transcript.clear()
                    committed = "CLEAR"

                elif lab == "SPEAK":
                    committed = "SPEAK"

                elif lab == "CONFIRM":
                    # CONFIRM doesn't modify transcript here; it signals end-of-message.
                    # Backend/UI decides what to do (e.g., move draft to history).
                    committed = "CONFIRM"

            # Reset candidate + enter cooldown
            self.candidate = None
            self.state = "COOLDOWN"
            self.cooldown = self.cfg.cooldown_frames

            # Reset counters for clean next word/command
            self.pred_window.clear()
            self.stable_counter = 0
            self.rest_counter = 0
            self.last_major_label = None

        if self.state == "COOLDOWN" and self.cooldown == 0:
            self.state = "IDLE"

        return {
            "live_label": major_label,
            "live_conf": avg_conf,
            "candidate": self.candidate,
            "transcript": " ".join(self.transcript),
            "committed": committed
        }

    def reset(self):
        """
        Hard reset for starting a fresh draft message.
        """
        self.transcript.clear()
        self.candidate = None
        self.pred_window.clear()
        self.state = "IDLE"
        self.cooldown = 0
        self.rest_counter = 0
        self.stable_counter = 0
        self.last_major_label = None
