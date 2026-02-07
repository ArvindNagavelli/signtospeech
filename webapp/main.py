import threading
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from runtime.hand_tracker import MediaPipeHandTracker
from runtime.features import build_feature_vector
from runtime.intent_engine import IntentEngine, IntentConfig
from runtime.inference import StaticModel

from webapp.state import SharedState
from webapp.video_stream import mjpeg_generator
from webapp.ws_utils import push_ws_updates
from webapp.history_store import HistoryStore

app = FastAPI()

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Persisted history (GDPR-friendly: local file, limited items)
HISTORY_PATH = ROOT / "data" / "history.json"
history_store = HistoryStore(file_path=HISTORY_PATH, max_items=200)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

state = SharedState()


def camera_worker(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try camera_index=1")

    tracker = MediaPipeHandTracker(model_complexity=0)

    cfg = IntentConfig()
    cfg.conf_word = 0.75
    cfg.stable_frames_word = 6
    cfg.conf_cmd = 0.85
    cfg.stable_frames_cmd = 18
    cfg.rest_frames = 4
    cfg.cooldown_frames = 8

    engine = IntentEngine(cfg)
    model = StaticModel()
    CONF_ACCEPT = 0.70

    # ✅ Load persisted history once at startup
    history = history_store.get_all()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)

        hf, _ = tracker.process(frame)
        hands_present = (hf.mask_left == 1 or hf.mask_right == 1)

        if hands_present:
            vec = build_feature_vector(hf)
            label, conf = model.predict(vec)
            if conf < CONF_ACCEPT:
                label = "UNKNOWN"
        else:
            label, conf = "NO_HANDS", 1.0

        ui = engine.update(label, conf, hands_present)

        committed = ui.get("committed")
        draft = ui.get("transcript", "")
        last_committed = committed or ""

        # ✅ CONFIRM: persist the draft into history.json
        if committed == "CONFIRM":
            msg = draft.strip()
            if msg:
                history_store.add(msg)          # write to disk
                history = history_store.get_all()  # refresh list from disk

            engine.reset()
            draft = ""
            last_committed = "CONFIRMED"

        # CLEAR: clear draft (engine already cleared transcript; draft reflect)
        if committed == "CLEAR":
            draft = ""

        cand = ui["candidate"]
        cand_text = cand[1] if cand else ""

        ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok2:
            continue

        state.update_from_frame(
            jpeg=jpg.tobytes(),
            live_label=ui["live_label"],
            live_conf=ui["live_conf"],
            candidate=cand_text,
            draft=draft,
            last_committed=last_committed,
            history=history,  # ✅ includes persisted history
        )

        time.sleep(0.01)


@app.on_event("startup")
def _startup():
    threading.Thread(target=camera_worker, kwargs={"camera_index": 0}, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def home():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/video")
def video():
    return StreamingResponse(
        mjpeg_generator(state.get_frame, fps=30),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await push_ws_updates(websocket, state.ws_payload, interval_s=0.1)
