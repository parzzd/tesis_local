# uvicorn app.server_inference:app --host 0.0.0.0 --port 8000
#
# Servidor de INFERENCIA (se despliega en RunPod con GPU).
#   - YOLO + LSTM + LGBM
#   - WebSocket /ws/stream/{cam_id}?src=... para streaming + detección
#   - /overlay/get y /overlay/set para togglear el render de poses
#   - SIN base de datos, SIN auth, SIN HTML
#
# El src de la cámara se recibe como query param (lo envía el frontend),
# así este servidor no necesita acceso a Supabase.

import os
import time
import base64
import asyncio
import logging
import threading
from typing import Set
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

try:
    cv2.setNumThreads(1)
except Exception:
    pass

# Evitar que TF pre-aloque toda la VRAM (~90%). Solo usa lo que el modelo necesita.
try:
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"[BOOT] No se pudo activar memory_growth de TF: {e}")

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import (
    SEQ_LEN, pool_frame_to_51, pool_scores,
    predict_window, load_artifacts,
)

load_dotenv()
log = logging.getLogger("inference")

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
MODELS_DIR   = Path(os.getenv("MODELS_DIR", "models_mix2"))
POSE_WEIGHTS = os.getenv("POSE_WEIGHTS", "yolo11s-pose.pt")

IMGSZ, CONF_POSE, IOU_POSE = 640, 0.25, 0.50
TOPK = 4

CONF_MIN     = 0.10
POOL_METHOD  = "topk"
TOPK_FRAC    = 0.20
FUSION_W     = float(os.getenv("FUSION_W", "0.50"))

VIDEO_MAX_SCORES = 900
DRAW_OVERLAY     = True

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Thread pool para operaciones bloqueantes (cv2, YOLO, Keras)
_executor = ThreadPoolExecutor(max_workers=2)

# Modelos pesados: se cargan bajo demanda.
ARTIFACTS = None
ARTIFACTS_LOCK = threading.Lock()


def get_artifacts():
    global ARTIFACTS
    if ARTIFACTS is not None:
        return ARTIFACTS

    with ARTIFACTS_LOCK:
        if ARTIFACTS is None:
            log.info("Cargando modelos bajo demanda desde %s", MODELS_DIR)
            ARTIFACTS = load_artifacts(MODELS_DIR, POSE_WEIGHTS)
    return ARTIFACTS


# ==========================================================
# APP
# ==========================================================
app = FastAPI(title="Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "inference", "ok": True}


@app.get("/health")
def health():
    return {"ok": True, "models_loaded": ARTIFACTS is not None}


# ── Overlay toggle ────────────────────────────────────────
@app.get("/overlay/get")
def overlay_get():
    return {"overlay": DRAW_OVERLAY}

@app.post("/overlay/set")
def overlay_set(payload: dict = Body(...)):
    global DRAW_OVERLAY
    DRAW_OVERLAY = bool(payload.get("overlay", True))
    return {"ok": True, "overlay": DRAW_OVERLAY}


# ==========================================================
# CAMERA WORKER
# ==========================================================
class CameraWorker:
    def __init__(self, cam_id: str, src: str):
        self.cam_id = cam_id
        self.src = src
        self.clients: Set[WebSocket] = set()
        self.running = False
        self.task = None

        self.win_feats = deque(maxlen=SEQ_LEN)
        self.video_scores = deque(maxlen=VIDEO_MAX_SCORES)
        self.on_state = False
        self.W: int = 0
        self.H: int = 0

    async def start(self):
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._loop())

    def _open_capture(self):
        src = int(self.src) if self.src.isdigit() else self.src
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return None
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap

    def _grab_and_infer(self, cap) -> dict | None:
        ok, frame = cap.read()
        if not ok:
            return None

        keras_model, mu, sd, thr_on, thr_off, lgbm, pose, stacker = get_artifacts()

        res = pose.predict(frame, imgsz=IMGSZ, conf=CONF_POSE, iou=IOU_POSE, verbose=False)[0]

        kps_f = None
        if res.keypoints is not None and res.keypoints.xy.shape[0] > 0:
            xy = res.keypoints.xy.cpu().numpy()
            c  = res.keypoints.conf.cpu().numpy()
            order = np.argsort(-c.mean(axis=1))
            P = min(len(order), TOPK)
            xy, c = xy[order[:P]], c[order[:P]]
            kps_f = np.concatenate([xy, c[..., None]], axis=-1).astype(np.float32)

        feat51 = pool_frame_to_51(kps_f, self.W, self.H)
        self.win_feats.append(feat51)

        p_win, p_vid = 0.0, 0.0

        if len(self.win_feats) == SEQ_LEN:
            Xw = np.stack(self.win_feats)
            p_win = predict_window(Xw, keras_model, mu, sd, lgbm=lgbm, fusion_w=FUSION_W, stacker=stacker)
            self.video_scores.append(p_win)
            p_vid = pool_scores(list(self.video_scores), pool=POOL_METHOD, topk_frac=TOPK_FRAC)

        fired_alert = False
        if not self.on_state and p_vid >= thr_on:
            self.on_state = True
            fired_alert = True
        elif self.on_state and p_vid <= thr_off:
            self.on_state = False

        try:
            shown = res.plot() if DRAW_OVERLAY else frame
        except Exception:
            shown = frame
        _, buf = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 65])
        jpg = base64.b64encode(buf).decode()

        now = time.time()
        return {
            "p_win": p_win,
            "p_vid": p_vid,
            "on": self.on_state,
            "ts": now,
            "jpg_b64": jpg,
            "fired_alert": fired_alert,
        }

    async def _loop(self):
        loop = asyncio.get_event_loop()

        cap = await loop.run_in_executor(_executor, self._open_capture)
        if cap is None:
            await self._broadcast({"type": "error", "msg": "No se pudo abrir la fuente de video"})
            self.running = False
            return

        try:
            while self.running and self.clients:
                result = await loop.run_in_executor(_executor, self._grab_and_infer, cap)
                if result is None:
                    await self._broadcast({"type": "error", "msg": "Fin del stream"})
                    break

                if result["fired_alert"]:
                    await self._broadcast({
                        "type": "alert",
                        "cam_id": self.cam_id,
                        "prob": result["p_vid"],
                        "ts": result["ts"],
                    })

                await self._broadcast({
                    "type": "frame",
                    "cam_id": self.cam_id,
                    "p_win": result["p_win"],
                    "p_vid": result["p_vid"],
                    "on": result["on"],
                    "ts": result["ts"],
                    "jpg_b64": result["jpg_b64"],
                })

                await asyncio.sleep(0.001)
        finally:
            await loop.run_in_executor(_executor, cap.release)
            self.running = False

    async def _broadcast(self, msg: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


# ── WebSocket endpoint ────────────────────────────────────
WORKERS: dict[str, CameraWorker] = {}


@app.websocket("/ws/stream/{cam_id}")
async def ws_stream(ws: WebSocket, cam_id: str, src: str = ""):
    """
    El frontend pasa el src como query param:
      wss://<runpod>/ws/stream/lobby?src=rtsp%3A%2F%2F...
    """
    await ws.accept()

    if not src:
        await ws.send_json({"type": "error", "msg": "missing src query param"})
        await ws.close()
        return

    worker = WORKERS.get(cam_id)
    if not worker or not worker.running:
        worker = CameraWorker(cam_id, src)
        WORKERS[cam_id] = worker
        await worker.start()

    worker.clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        worker.clients.discard(ws)
