"""Lightweight entry point for Render deployment. Zero heavy dependencies."""
import sys
import os
import asyncio
import json
import math
import random
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

TEMPLATES_DIR = Path(__file__).parent / "reverse_bci" / "ui" / "templates"
RESULTS_PATH = Path(__file__).parent / "checkpoints" / "experiments" / "all_results.json"

CHANNEL_NAMES = ["Cz", "F7", "Fz", "C3", "T8", "F8", "C4", "T7"]
WORDS = ["UP", "DOWN", "LEFT", "RIGHT"]

app = FastAPI(title="TRIBE BCI")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/results")
async def get_results():
    if RESULTS_PATH.exists():
        return JSONResponse(json.loads(RESULTS_PATH.read_text()))
    return JSONResponse({
        "channel_downsample": {
            "4ch_speech": {"mean": 28.0, "std": 3.7, "p": 0.18},
            "8ch_speech": {"mean": 30.5, "std": 4.8, "p": 0.045},
            "16ch_speech": {"mean": 25.5, "std": 8.0, "p": 0.46},
            "128ch_full": {"mean": 28.0, "std": 7.5, "p": 0.18},
        }
    })


@app.get("/api/hardware")
async def get_hardware():
    positions = [
        (0.0, 0.0), (-0.55, -0.35), (0.0, -0.35), (-0.45, 0.0),
        (0.7, 0.0), (0.55, -0.35), (0.45, 0.0), (-0.7, 0.0),
    ]
    regions = [
        "Central (vertex)", "Left frontal (Broca's)", "Frontal midline",
        "Left central (motor)", "Right temporal", "Right frontal",
        "Right central (motor)", "Left temporal",
    ]
    channels = [
        {"name": CHANNEL_NAMES[i], "brain_region": regions[i],
         "position": {"x": positions[i][0], "y": positions[i][1]}}
        for i in range(8)
    ]
    return JSONResponse({
        "channel_count": 8, "channels": channels,
        "cost_comparison": [
            {"name": "TRIBE BCI", "channels": 8, "cost": 800, "accuracy": 30.5},
            {"name": "Clinical EEG", "channels": 128, "cost": 50000, "accuracy": 28.0},
        ],
    })


@app.get("/api/status")
async def get_status():
    return JSONResponse({"status": "online", "eeg_source": "synthetic", "channels": 8})


@app.websocket("/ws/eeg")
async def eeg_stream(websocket: WebSocket):
    await websocket.accept()
    t = 0.0
    interval = 1.0 / 30.0
    last_pred = 0.0

    try:
        while True:
            values = []
            for ch in range(8):
                phase = ch * 0.4
                alpha = 15.0 * math.sin(2 * math.pi * 10.0 * t + phase)
                beta = 5.0 * math.sin(2 * math.pi * 20.0 * t + phase * 1.3)
                gamma = 2.0 * math.sin(2 * math.pi * 38.0 * t + phase * 0.7)
                noise = (random.random() - 0.5) * 8.0
                drift = 5.0 * math.sin(2 * math.pi * 0.3 * t + ch)
                values.append(round(alpha + beta + gamma + noise + drift, 2))

            pred = None
            if t - last_pred >= 3.0 + random.random() * 2.0:
                last_pred = t
                top = random.uniform(0.28, 0.65)
                remaining = 1.0 - top
                probs = [0.0] * 4
                winner = random.randint(0, 3)
                probs[winner] = round(top, 3)
                for i in range(4):
                    if i != winner:
                        share = random.random() * remaining * 0.6
                        probs[i] = round(share, 3)
                        remaining -= share
                pred = {"word": WORDS[winner], "conf": round(top, 3), "probs": probs}

            await websocket.send_json({
                "t": round(t, 4), "ch": values,
                "labels": CHANNEL_NAMES, "pred": pred,
            })
            t += interval
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
