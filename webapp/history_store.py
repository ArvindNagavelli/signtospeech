import json
from pathlib import Path
from typing import List


class HistoryStore:
    """
    Simple local JSON-backed store for confirmed messages.
    GDPR-friendly: keep minimal data, for minimal time, locally.
    """

    def __init__(self, file_path: Path, max_items: int = 200):
        self.file_path = file_path
        self.max_items = max_items
        self._history: List[str] = []
        self.load()

    def load(self) -> None:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._history = [str(x) for x in data][-self.max_items :]
            except Exception:
                # If file is corrupted, start fresh
                self._history = []

    def save(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self._history[-self.max_items :], f, ensure_ascii=False, indent=2)

    def add(self, message: str) -> None:
        msg = (message or "").strip()
        if not msg:
            return
        self._history.append(msg)
        self._history = self._history[-self.max_items :]
        self.save()

    def clear(self) -> None:
        self._history = []
        self.save()

    def get_all(self) -> List[str]:
        return list(self._history)
