# uvicorn app.server:app --host 0.0.0.0 --port 8000
import os, time, base64, asyncio, logging, threading
from typing import Dict, Set
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import Base, engine, SessionLocal
from app.models import User, AccessLog, CameraAction, AlertLog, Camera
from app.schemas import (
    LoginRequest, RegisterRequest, CameraConfig, AlertDecision,
)
from app.utils import make_salt, hash_password, verify_password
from app.crud import (
    create_user, get_user_by_email, add_access_log,
    add_camera_action, add_alert_log,
)
from app.pipeline import (
    SEQ_LEN, pool_frame_to_51, frame_visible, pool_scores,
    predict_window, load_artifacts,
)

load_dotenv()
log = logging.getLogger("server")

# ==========================================================
# CONFIGURACIÓN (desde .env o valores por defecto)
# ==========================================================
ROOT_DIR   = Path(__file__).resolve().parent
STATIC_DIR = (ROOT_DIR / "static").resolve()
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models_mix"))

POSE_WEIGHTS = os.getenv("POSE_WEIGHTS", "yolo11s-pose.pt")
IMGSZ, CONF_POSE, IOU_POSE = 640, 0.25, 0.50
TOPK = 4

CONF_MIN     = 0.10
POOL_METHOD  = "topk"
TOPK_FRAC    = 0.20
FUSION_W     = float(os.getenv("FUSION_W", "0.50"))

VIDEO_MAX_SCORES = 900
DRAW_OVERLAY     = True

LOG_RETENTION_DAYS = 7

BOSS_CODE = os.getenv("BOSS_CODE", "20261")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

FAILED_ATTEMPTS: dict[str, dict] = {}
MAX_ATTEMPTS       = 3
BLOCK_TIME_MINUTES = 5

# Thread pool para operaciones bloqueantes (cv2, YOLO, Keras)
_executor = ThreadPoolExecutor(max_workers=4)


# ==========================================================
# LIMPIEZA PERIÓDICA DE LOGS
# ==========================================================
def _cleanup_logs():
    while True:
        try:
            db = SessionLocal()
            limit = datetime.utcnow() - timedelta(days=LOG_RETENTION_DAYS)
            db.query(AccessLog).filter(AccessLog.timestamp < limit).delete()
            db.query(CameraAction).filter(CameraAction.timestamp < limit).delete()
            db.commit()
            db.close()
        except Exception as e:
            log.warning("Error al borrar logs: %s", e)
        time.sleep(24 * 3600)

threading.Thread(target=_cleanup_logs, daemon=True).start()


# ==========================================================
# BASE DE DATOS
# ==========================================================
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==========================================================
# CARGAR MODELOS (LSTM + LGBM + YOLO)
# ==========================================================
KERAS, MU, SD, THR_ON, THR_OFF, LGBM, POSE, STACKER = load_artifacts(MODELS_DIR, POSE_WEIGHTS)


# ==========================================================
# APP
# ==========================================================
app = FastAPI(title="Sistema de Videovigilancia")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Páginas ───────────────────────────────────────────────
@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "login.html")

@app.get("/dashboard")
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/admin")
def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")


# ── Overlay toggle ────────────────────────────────────────
@app.get("/overlay/get")
def overlay_get():
    return {"overlay": DRAW_OVERLAY}

@app.post("/overlay/set")
def overlay_set(payload: dict = Body(...)):
    global DRAW_OVERLAY
    DRAW_OVERLAY = bool(payload.get("overlay", True))
    return {"ok": True, "overlay": DRAW_OVERLAY}


# ── Auth ──────────────────────────────────────────────────
@app.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    email = req.email.lower().strip()
    now = datetime.now()

    info = FAILED_ATTEMPTS.get(email, {"count": 0, "blocked_until": None})
    if info["blocked_until"] and now < info["blocked_until"]:
        return JSONResponse({"error": "Bloqueado"}, status_code=403)

    user = get_user_by_email(db, email)
    if not user or not verify_password(req.password, user.salt, user.password):
        info["count"] += 1
        if info["count"] >= MAX_ATTEMPTS:
            info["blocked_until"] = now + timedelta(minutes=BLOCK_TIME_MINUTES)
        FAILED_ATTEMPTS[email] = info
        return JSONResponse({"error": "Credenciales inválidas"}, status_code=401)

    FAILED_ATTEMPTS[email] = {"count": 0, "blocked_until": None}
    add_access_log(db, user.id)

    return {
        "access_token": "token_valido",
        "email": email,
        "nombre": user.nombre,
        "apellido": user.apellido,
        "charge": user.charge,
    }


@app.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    email = req.email.lower().strip()

    if req.charge == "jefe" and req.codigo != BOSS_CODE:
        return JSONResponse({"error": "Código incorrecto"}, status_code=403)

    if get_user_by_email(db, email):
        return JSONResponse({"error": "Usuario existente"}, status_code=409)

    create_user(db, req.nombre, req.apellido, email, req.password, req.charge)
    return {"ok": True}


# ── Admin endpoints ───────────────────────────────────────
@app.get("/admin/users")
def admin_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{
        "id": u.id, "nombre": u.nombre, "apellido": u.apellido,
        "email": u.username, "charge": u.charge,
    } for u in users]


@app.get("/admin/logs/access")
def admin_access_logs(db: Session = Depends(get_db)):
    logs = db.query(AccessLog).order_by(AccessLog.timestamp.desc()).all()
    return [{
        "usuario": f"{log.user.nombre} {log.user.apellido}" if log.user else "—",
        "cargo": log.user.charge if log.user else "—",
        "timestamp": log.timestamp.timestamp(),
    } for log in logs]


@app.get("/admin/logs/cameras")
def admin_camera_logs(db: Session = Depends(get_db)):
    logs = db.query(CameraAction).order_by(CameraAction.timestamp.desc()).all()
    return [{
        "usuario": f"{log.user.nombre} {log.user.apellido}" if log.user else "—",
        "cam_id": log.cam_id,
        "action": log.action,
        "timestamp": log.timestamp.timestamp(),
    } for log in logs]


@app.get("/admin/logs/alerts")
def admin_alert_logs(db: Session = Depends(get_db)):
    logs = db.query(AlertLog).order_by(AlertLog.timestamp.desc()).all()
    return [{
        "cam_id": l.cam_id,
        "prob": l.prob,
        "timestamp": l.timestamp.timestamp(),
    } for l in logs]


@app.delete("/admin/logs/access/clear")
def clear_access_logs(db: Session = Depends(get_db)):
    db.query(AccessLog).delete()
    db.commit()
    return {"ok": True, "msg": "Historial de accesos eliminado"}

@app.delete("/admin/logs/actions/clear")
def clear_action_logs(db: Session = Depends(get_db)):
    db.query(CameraAction).delete()
    db.commit()
    return {"ok": True, "msg": "Historial de acciones eliminado"}

@app.delete("/admin/logs/alerts/clear")
def clear_alert_logs(db: Session = Depends(get_db)):
    db.query(AlertLog).delete()
    db.commit()
    return {"ok": True, "msg": "Alertas eliminadas"}


# ==========================================================
# CÁMARAS  –  persistidas en DB, cargadas al inicio
# ==========================================================
def _load_cameras_from_db() -> Dict[str, CameraConfig]:
    db = SessionLocal()
    try:
        rows = db.query(Camera).all()
        return {r.cam_id: CameraConfig(cam_id=r.cam_id, src=r.src) for r in rows}
    finally:
        db.close()

CAMERAS: Dict[str, CameraConfig] = _load_cameras_from_db()


@app.post("/cameras")
def add_camera(cfg: CameraConfig, db: Session = Depends(get_db)):
    CAMERAS[cfg.cam_id] = cfg

    existing = db.query(Camera).filter(Camera.cam_id == cfg.cam_id).first()
    if existing:
        existing.src = cfg.src
    else:
        db.add(Camera(cam_id=cfg.cam_id, src=cfg.src))
    db.commit()

    add_camera_action(db, 1, cfg.cam_id, "add")
    return {"ok": True}


@app.delete("/cameras/{cam_id}")
def delete_camera(cam_id: str, db: Session = Depends(get_db)):
    CAMERAS.pop(cam_id, None)

    db.query(Camera).filter(Camera.cam_id == cam_id).delete()
    db.commit()

    # Detener worker si está corriendo
    worker = WORKERS.pop(cam_id, None)
    if worker:
        worker.running = False

    add_camera_action(db, 1, cam_id, "delete")
    return {"ok": True}


@app.get("/cameras")
def list_cameras():
    return list(CAMERAS.values())


# ==========================================================
# CAMERA WORKER  (WebSocket streaming + inferencia LSTM+LGBM)
#
# Las operaciones bloqueantes (cv2, YOLO, Keras) se ejecutan
# en un ThreadPoolExecutor para NO bloquear el event loop.
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

    # ── Parte bloqueante (corre en thread) ────────────────
    def _open_capture(self):
        src = int(self.src) if self.src.isdigit() else self.src
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return None
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap

    def _grab_and_infer(self, cap) -> dict | None:
        """Lee un frame, ejecuta YOLO + inferencia. Retorna dict o None si terminó."""
        ok, frame = cap.read()
        if not ok:
            return None

        # Pose estimation
        res = POSE.predict(frame, imgsz=IMGSZ, conf=CONF_POSE, iou=IOU_POSE, verbose=False)[0]

        kps_f = None
        if res.keypoints is not None and res.keypoints.xy.shape[0] > 0:
            xy = res.keypoints.xy.cpu().numpy()
            c  = res.keypoints.conf.cpu().numpy()
            order = np.argsort(-c.mean(axis=1))
            P = min(len(order), TOPK)
            xy, c = xy[order[:P]], c[order[:P]]
            kps_f = np.concatenate([xy, c[..., None]], axis=-1).astype(np.float32)

        # Features
        feat51 = pool_frame_to_51(kps_f, self.W, self.H)
        self.win_feats.append(feat51)

        p_win, p_vid = 0.0, 0.0

        if len(self.win_feats) == SEQ_LEN:
            Xw = np.stack(self.win_feats)
            p_win = predict_window(Xw, KERAS, MU, SD, lgbm=LGBM, fusion_w=FUSION_W, stacker=STACKER)
            self.video_scores.append(p_win)
            p_vid = pool_scores(list(self.video_scores), pool=POOL_METHOD, topk_frac=TOPK_FRAC)

        # Histéresis
        fired_alert = False
        if not self.on_state and p_vid >= THR_ON:
            self.on_state = True
            fired_alert = True
        elif self.on_state and p_vid <= THR_OFF:
            self.on_state = False

        # Render JPEG
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

    # ── Loop async (delega al thread pool) ────────────────
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
WORKERS: Dict[str, CameraWorker] = {}


@app.websocket("/ws/stream/{cam_id}")
async def ws_stream(ws: WebSocket, cam_id: str):
    await ws.accept()

    if cam_id not in CAMERAS:
        await ws.send_json({"type": "error", "msg": "cam-not-found"})
        await ws.close()
        return

    worker = WORKERS.get(cam_id)
    if not worker or not worker.running:
        worker = CameraWorker(cam_id, CAMERAS[cam_id].src)
        WORKERS[cam_id] = worker
        await worker.start()

    worker.clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        worker.clients.discard(ws)
        # Si no quedan clientes, el worker se detiene solo
        # (la condición `self.clients` en _loop se encarga)


# ── Alertas (aceptar/rechazar) ────────────────────────────
@app.post("/alerts/save")
def save_alert(decision: AlertDecision, db: Session = Depends(get_db)):
    if decision.accepted:
        add_alert_log(db, decision.cam_id, decision.prob)
        return {"ok": True, "msg": "Alerta guardada"}
    return {"ok": True, "msg": "Alerta descartada"}
