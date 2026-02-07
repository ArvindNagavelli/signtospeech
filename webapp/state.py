# webapp/state.py
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Optional, Dict, Any


@dataclass
class SharedState:
    lock: Lock = field(default_factory=Lock)

    latest_jpeg: Optional[bytes] = None
    latest_live_label: str = "NO_HANDS"
    latest_live_conf: float = 1.0
    latest_candidate: str = ""
    draft_sentence: str = ""
    last_committed: str = ""
    history: List[str] = field(default_factory=list)

    def update_from_frame(
        self,
        jpeg: bytes,
        live_label: str,
        live_conf: float,
        candidate: str,
        draft: str,
        last_committed: str,
        history: List[str],
    ) -> None:
        with self.lock:
            self.latest_jpeg = jpeg
            self.latest_live_label = live_label
            self.latest_live_conf = live_conf
            self.latest_candidate = candidate
            self.draft_sentence = draft
            self.last_committed = last_committed
            self.history = history

    def get_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def ws_payload(self) -> Dict[str, Any]:
        with self.lock:
            live_label = self.latest_live_label
            if live_label == "NO_HANDS":
                live_label = "—"

            return {
                "live_label": live_label,
                "live_conf": round(float(self.latest_live_conf), 3),
                "candidate": self.latest_candidate if self.latest_candidate else "—",
                "draft": self.draft_sentence,
                "last_committed": self.last_committed if self.last_committed else "—",
                "history": self.history[-10:]
            }
