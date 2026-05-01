import traceback
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.controllers.recognition_controller import (
    predict_alphabet,
    predict_number,
    predict_word
)

router = APIRouter()


@router.websocket("/ws/predict/alphabet")
async def alphabet_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                result = predict_alphabet(data)
            except Exception as exc:
                traceback.print_exc()
                result = {"sign": "...", "confidence": 0.0, "landmarks": None,
                          "committed": False, "hold_progress": 0.0,
                          "error": str(exc)}
            await websocket.send_json(result)
    except WebSocketDisconnect:
        print("Alphabet WS disconnected")


@router.websocket("/ws/predict/number")
async def number_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                result = predict_number(data)
            except Exception as exc:
                traceback.print_exc()
                result = {"sign": "...", "confidence": 0.0, "landmarks": None,
                          "committed": False, "hold_progress": 0.0,
                          "error": str(exc)}
            await websocket.send_json(result)
    except WebSocketDisconnect:
        print("Number WS disconnected")


@router.websocket("/ws/predict/word")
async def word_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                result = predict_word(data)
            except Exception as exc:
                traceback.print_exc()
                result = {"sign": "...", "confidence": 0.0, "ready": False,
                          "top3": [], "collecting": False, "frames": 0,
                          "error": str(exc)}
            await websocket.send_json(result)
    except WebSocketDisconnect:
        print("Word WS disconnected")