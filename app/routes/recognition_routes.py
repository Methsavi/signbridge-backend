from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.controllers.recognition_controller import process_frame
from collections import Counter, deque
import asyncio

router = APIRouter()


@router.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    print("🔄 Client connecting to WebSocket...")
    await websocket.accept()
    print("✅ Client connected!")

    # --- FASTER BUFFER ---
    # Only wait for 5 frames (approx 0.5 seconds) instead of 10
    history = deque(maxlen=5)

    try:
        while True:
            # 1. Receive Image
            data = await websocket.receive_text()

            # 2. Process it
            result = process_frame(data)

            # 3. Add to History
            if "sign" in result:
                history.append(result["sign"])

            # 4. Smart Voting
            if len(history) == 5:
                counts = Counter(history)
                most_common_sign, frequency = counts.most_common(1)[0]

                # If 3 out of 5 frames agree, update the text
                if frequency >= 3:
                    final_result = {
                        "sign": most_common_sign,
                        "confidence": result.get("confidence", 0)
                    }
                    await websocket.send_json(final_result)
                else:
                    # Even if unstable, send the latest guess so UI doesn't freeze
                    # But maybe mark confidence lower
                    result["confidence"] = result.get("confidence", 0) - 20
                    await websocket.send_json(result)
            else:
                # Buffer filling up, send raw data immediately
                await websocket.send_json(result)

            # Tiny pause to prevent CPU overheating
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        print(f"⚠️ WebSocket Error: {e}")