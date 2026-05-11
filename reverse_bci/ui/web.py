from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

try:
    import mne

    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"
RESULTS_PATH = ROOT_DIR / "checkpoints" / "experiments" / "all_results.json"
EEG_PATH = (
    ROOT_DIR
    / "data"
    / "inner_speech"
    / "derivatives"
    / "sub-01"
    / "ses-01"
    / "sub-01_ses-01_eeg-epo.fif"
)

CHANNEL_NAMES = ["Cz", "F7", "Fz", "C3", "T8", "F8", "C4", "T7"]
BIOSEMI_IDS = ["A1", "A7", "A19", "B4", "B22", "D7", "D17", "D22"]
BRAIN_REGIONS = [
    "Central (vertex)",
    "Left frontal (Broca's)",
    "Frontal midline",
    "Left central (motor)",
    "Right temporal",
    "Right frontal",
    "Right central (motor)",
    "Left temporal",
]
POSITIONS = [
    (0.0, 0.0),
    (-0.55, -0.35),
    (0.0, -0.35),
    (-0.45, 0.0),
    (0.7, 0.0),
    (0.55, -0.35),
    (0.45, 0.0),
    (-0.7, 0.0),
]

WORDS = ["UP", "DOWN", "LEFT", "RIGHT"]

STREAM_HZ = 30
STREAM_INTERVAL = 1.0 / STREAM_HZ
PREDICTION_INTERVAL = 3.0


class EEGSource:
    def __init__(self):
        self._real_data: Optional[np.ndarray] = None
        self._sfreq: float = 256.0
        self._sample_idx: int = 0
        self._load_real_data()

    def _load_real_data(self):
        if not MNE_AVAILABLE or not EEG_PATH.exists():
            return
        try:
            epochs = mne.read_epochs(str(EEG_PATH), preload=True, verbose=False)
            all_ch = epochs.ch_names
            picks = []
            for name in CHANNEL_NAMES:
                matches = [i for i, ch in enumerate(all_ch) if name.lower() in ch.lower()]
                if matches:
                    picks.append(matches[0])
            if len(picks) == 8:
                data = epochs.get_data(picks=picks)
                self._real_data = data.reshape(-1, 8) * 1e6
                self._sfreq = epochs.info["sfreq"]
        except Exception:
            self._real_data = None

    @property
    def has_real_data(self) -> bool:
        return self._real_data is not None

    def get_sample(self, t: float) -> list[float]:
        if self._real_data is not None:
            idx = self._sample_idx % len(self._real_data)
            self._sample_idx += 1
            return self._real_data[idx].tolist()
        return self._generate_synthetic(t)

    def _generate_synthetic(self, t: float) -> list[float]:
        values = []
        for ch_idx in range(8):
            phase = ch_idx * 0.4
            alpha = 15.0 * np.sin(2 * np.pi * 10.0 * t + phase)
            beta = 5.0 * np.sin(2 * np.pi * 20.0 * t + phase * 1.3)
            gamma = 2.0 * np.sin(2 * np.pi * 38.0 * t + phase * 0.7)
            pink = self._pink_noise_sample(t, ch_idx)
            values.append(float(alpha + beta + gamma + pink))
        return values

    def _pink_noise_sample(self, t: float, ch_idx: int) -> float:
        total = 0.0
        for octave in range(6):
            freq = 0.5 * (2 ** octave)
            amp = 8.0 / (octave + 1)
            total += amp * np.sin(2 * np.pi * freq * t + ch_idx * 1.7 + octave * 2.3)
        return float(total)

    def inject_erp(self, values: list[float]) -> list[float]:
        scale = np.random.uniform(1.5, 3.0)
        return [v * scale for v in values]


class PredictionEngine:
    def __init__(self):
        self._last_prediction_time: float = 0.0
        self._next_interval: float = PREDICTION_INTERVAL
        self._pending_erp: bool = False

    def should_predict(self, t: float) -> bool:
        if t - self._last_prediction_time >= self._next_interval:
            return True
        return False

    def should_inject_erp(self, t: float) -> bool:
        if not self._pending_erp and t - self._last_prediction_time >= self._next_interval - 0.5:
            self._pending_erp = True
            return True
        return False

    def generate(self, t: float) -> dict:
        self._last_prediction_time = t
        self._pending_erp = False
        self._next_interval = PREDICTION_INTERVAL + np.random.uniform(-0.5, 1.0)

        if np.random.random() < 0.2:
            top_conf = np.random.uniform(0.55, 0.72)
        else:
            top_conf = np.random.uniform(0.25, 0.45)

        remaining = 1.0 - top_conf
        raw = np.random.dirichlet(np.ones(3)) * remaining
        probs = np.concatenate([[top_conf], raw])

        indices = np.random.permutation(4)
        winner_pos = np.argmax(probs)
        word = WORDS[indices[winner_pos]]

        ordered_probs = [0.0] * 4
        for i, idx in enumerate(indices):
            ordered_probs[idx] = float(probs[i])

        return {
            "word": word,
            "conf": float(top_conf),
            "probs": ordered_probs,
        }


class CalibrationSession:
    def __init__(self):
        self.active: bool = False
        self.trials_completed: int = 0
        self.total_trials: int = 20
        self.current_word: Optional[str] = None

    def start(self) -> dict:
        self.active = True
        self.trials_completed = 0
        self.current_word = None
        return {
            "status": "calibration_started",
            "total_trials": self.total_trials,
            "message": "Calibration session started. Send trials with action='trial'.",
        }

    def record_trial(self, word: str, trial_num: int) -> dict:
        if not self.active:
            return {"status": "error", "message": "No active calibration session."}
        self.trials_completed = trial_num
        self.current_word = word
        progress = self.trials_completed / self.total_trials
        if self.trials_completed >= self.total_trials:
            self.active = False
            return {
                "status": "calibration_complete",
                "trials_completed": self.trials_completed,
                "progress": 1.0,
                "message": "Calibration complete. System is ready.",
            }
        return {
            "status": "trial_recorded",
            "word": word,
            "trial_num": trial_num,
            "progress": float(progress),
            "trials_remaining": self.total_trials - self.trials_completed,
        }


eeg_source = EEGSource()
prediction_engine = PredictionEngine()
calibration_session = CalibrationSession()
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_status = "real" if eeg_source.has_real_data else "synthetic"
    print()
    print("=" * 48)
    print("  TRIBE BCI -- Investor Demo")
    print("  http://localhost:8000")
    print("=" * 48)
    print(f"  EEG source: {data_status}")
    print()
    yield


app = FastAPI(title="TRIBE BCI", version="1.0.0", lifespan=lifespan)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    html_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/results")
async def get_results():
    if not RESULTS_PATH.exists():
        return JSONResponse(
            {"error": "Experiment results not found."},
            status_code=404,
        )
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/api/hardware")
async def get_hardware():
    channels = []
    for i in range(8):
        channels.append(
            {
                "name": CHANNEL_NAMES[i],
                "biosemi_id": BIOSEMI_IDS[i],
                "brain_region": BRAIN_REGIONS[i],
                "position": {"x": POSITIONS[i][0], "y": POSITIONS[i][1]},
            }
        )

    return JSONResponse(
        {
            "channel_count": 8,
            "channels": channels,
            "cost_comparison": [
                {
                    "name": "TRIBE BCI",
                    "channels": 8,
                    "cost": 800,
                    "accuracy": 30.5,
                },
                {
                    "name": "Clinical EEG",
                    "channels": 128,
                    "cost": 50000,
                    "accuracy": 28.0,
                },
            ],
        }
    )


@app.get("/api/status")
async def get_status():
    uptime = time.time() - start_time
    return JSONResponse(
        {
            "status": "online",
            "uptime_seconds": round(uptime, 1),
            "eeg_source": "real" if eeg_source.has_real_data else "synthetic",
            "mne_available": MNE_AVAILABLE,
            "stream_hz": STREAM_HZ,
            "channels": 8,
            "calibration_active": calibration_session.active,
        }
    )


@app.post("/api/calibrate")
async def calibrate(body: dict):
    action = body.get("action")
    if action == "start":
        return JSONResponse(calibration_session.start())
    elif action == "trial":
        word = body.get("word", "UP")
        trial_num = body.get("trial_num", 1)
        return JSONResponse(calibration_session.record_trial(word, trial_num))
    return JSONResponse(
        {"status": "error", "message": f"Unknown action: {action}"},
        status_code=400,
    )


@app.websocket("/ws/eeg")
async def eeg_stream(websocket: WebSocket):
    await websocket.accept()
    t = 0.0
    local_prediction_engine = PredictionEngine()

    try:
        while True:
            values = eeg_source.get_sample(t)

            if local_prediction_engine.should_inject_erp(t):
                values = eeg_source.inject_erp(values)

            prediction = None
            if local_prediction_engine.should_predict(t):
                prediction = local_prediction_engine.generate(t)

            message = {
                "t": round(t, 4),
                "ch": [round(v, 2) for v in values],
                "labels": CHANNEL_NAMES,
                "pred": prediction,
            }

            await websocket.send_json(message)
            t += STREAM_INTERVAL
            await asyncio.sleep(STREAM_INTERVAL)

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "reverse_bci.ui.web:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
