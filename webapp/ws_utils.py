# webapp/ws_utils.py
import asyncio
from typing import Callable, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect


async def push_ws_updates(
    websocket: WebSocket,
    get_payload: Callable[[], Dict[str, Any]],
    interval_s: float = 0.1
) -> None:
    """
    Generic websocket sender loop (safe, reusable).
    """
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(get_payload())
            await asyncio.sleep(interval_s)
    except WebSocketDisconnect:
        return
