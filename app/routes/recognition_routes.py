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
            result = predict_alphabet(data)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        print("Alphabet WS disconnected")


@router.websocket("/ws/predict/number")
async def number_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            result = predict_number(data)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        print("Number WS disconnected")


@router.websocket("/ws/predict/word")
async def word_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            result = predict_word(data)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        print("Word WS disconnected")