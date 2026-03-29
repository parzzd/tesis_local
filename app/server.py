# uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
import time, json, base64, asyncio
from typing import Dict, Set
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import joblib
import hashlib
from tensorflow.keras.models import load_model

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
import threading


#PENDIENTES
#agregar dashboard de alertas en administrador
#emitir alerta simulado, guardar en base de datos/rechazar (camara,operador,timestamp,probabilidad)
######################

from sqlalchemy.orm import Session
from app.database import Base, engine, SessionLocal
from app.models import *
LOG_RETENTION_DAYS = 7 #variar el dia para limpieza de logs

def cleanup_logs_periodically():
    while True:
        try:
            db = SessionLocal()

            limit_ts = time.time() - (LOG_RETENTION_DAYS * 24 * 3600)

            db.query(AccessLog).filter(AccessLog.timestamp < limit_ts).delete()

            db.query(CameraAction).filter(CameraAction.timestamp < limit_ts).delete()

            db.commit()
            db.close()

        except Exception as e:
            print("Error al borrar logs:", e)

        time.sleep(24 * 3600)

threading.Thread(target=cleanup_logs_periodically, daemon=True).start()


Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(p: str):
    return hashlib.sha256(p.encode()).hexdigest()


# ==========================================================
# CONFIGURACIÓN GLOBAL
# ==========================================================
ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = (ROOT_DIR / "static").resolve()

KERAS_MODEL = Path("models_mix/mix_cnn_lstm_T32_F51.keras")
NORM_STATS  = Path("models_mix/mix_cnn_lstm_T32_F51_norm_stats.npz")
THRESH_JSON = Path("models_mix/mix_cnn_lstm_T32_F51_threshold.json")
LGBM_PKL    = Path("models_mix/lgbm_model.pkl")

POSE_WEIGHTS = "yolo11s-pose.pt"
IMGSZ, CONF_POSE, IOU_POSE = 640, 0.25, 0.50
TOPK = 4

SEQ_LEN = 32
CONF_MIN = 0.10
POOL_METHOD = "topk"
TOPK_FRAC = 0.20
HYST_GAP = 0.10

VIDEO_MAX_SCORES = 900
SEND_FPS = 10
JPEG_QUALITY = 65

DRAW_OVERLAY = True

FAILED_ATTEMPTS: dict[str, dict] = {}
MAX_ATTEMPTS = 3
BLOCK_TIME_MINUTES = 5


# ==========================================================
# PIPELINE UTILS
# ==========================================================
def pool_frame_to_51(kps_f, W, H):
    out = np.zeros((17,3), np.float32)
    if kps_f is None or kps_f.size == 0:
        return out.reshape(-1)
    conf_j = np.nan_to_num(kps_f[...,2])
    for j in range(17):
        idx = int(np.argmax(conf_j[:,j]))
        x,y,c = kps_f[idx,j]
        if c>0:
            out[j,0] = x/W
            out[j,1] = y/H
            out[j,2] = c
    return out.reshape(-1)

def frame_visible(kps_f, conf_min=CONF_MIN):
    if kps_f is None: return False
    return bool((np.nan_to_num(kps_f[...,2]) >= conf_min).any())

def pool_scores(scores, pool="topk", topk_frac=0.2):
    if not scores: return 0.0
    arr = np.array(scores)
    if pool == "max": return float(arr.max())
    if pool == "mean": return float(arr.mean())
    k = max(1, int(len(arr)*topk_frac))
    return float(np.partition(arr, -k)[-k:].mean())



def load_artifacts():
    keras_model = load_model(str(KERAS_MODEL), compile=False)
    stats = np.load(NORM_STATS)
    mu, sd = stats["mean"].astype("float32"), stats["std"].astype("float32")

    thr = 0.5
    if THRESH_JSON.exists():
        thr = float(json.loads(Path(THRESH_JSON).read_text())["best_threshold"])

    lgbm = joblib.load(LGBM_PKL) if LGBM_PKL.exists() else None
    pose = YOLO(POSE_WEIGHTS)

    return keras_model, mu, sd, thr, lgbm, pose

KERAS, MU, SD, THR_ON, LGBM, POSE = load_artifacts()
THR_OFF = THR_ON - HYST_GAP



app = FastAPI(title="vigilancia sistema")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "login.html")

@app.get("/dashboard")
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/admin")
def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")



@app.get("/overlay/get")
def overlay_get():
    return {"overlay": DRAW_OVERLAY}

@app.post("/overlay/set")
def overlay_set(payload: dict = Body(...)):
    global DRAW_OVERLAY
    DRAW_OVERLAY = bool(payload.get("overlay", True))
    return {"ok": True, "overlay": DRAW_OVERLAY}


@app.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    email = req.email.lower().strip()
    now = datetime.now()

    info = FAILED_ATTEMPTS.get(email, {"count":0, "blocked_until":None})

    if info["blocked_until"] and now < info["blocked_until"]:
        return JSONResponse({"error":"Bloqueado"}, status_code=403)

    user = db.query(User).filter(User.username==email).first()
    if not user or hash_password(req.password) != user.password:
        info["count"] += 1
        if info["count"] >= MAX_ATTEMPTS:
            info["blocked_until"] = now + timedelta(minutes=BLOCK_TIME_MINUTES)
        FAILED_ATTEMPTS[email] = info
        return JSONResponse({"error":"Credenciales inválidas"}, status_code=401)

    FAILED_ATTEMPTS[email] = {"count":0, "blocked_until":None}

    db.add(AccessLog(user_id=user.id))
    db.commit()

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

    if req.charge == "jefe" and req.codigo != "20261":
        return JSONResponse({"error": "Código incorrecto"}, status_code=403)

    if db.query(User).filter(User.username==email).first():
        return JSONResponse({"error": "Usuario existente"}, status_code=409)

    user = User(
        nombre=req.nombre,
        apellido=req.apellido,
        username=email,
        password=hash_password(req.password),
        charge=req.charge
    )
    db.add(user)
    db.commit()

    return {"ok": True}


@app.get("/admin/users")
def admin_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{
        "id": u.id,
        "nombre": u.nombre,
        "apellido": u.apellido,
        "email": u.username,
        "charge": u.charge
    } for u in users]


@app.get("/admin/logs/access")
def admin_access_logs(db: Session = Depends(get_db)):
    logs = db.query(AccessLog).all()
    out = []
    for log in logs:
        u = db.query(User).filter(User.id == log.user_id).first()
        out.append({
            "usuario": f"{u.nombre} {u.apellido}" if u else "—",
            "cargo": u.charge if u else "—",
            "timestamp": log.timestamp,
        })
    return out


@app.get("/admin/logs/cameras")
def admin_camera_logs(db: Session = Depends(get_db)):
    logs = db.query(CameraAction).all()
    out = []
    for log in logs:
        u = db.query(User).filter(User.id==log.user_id).first()
        out.append({
            "usuario": f"{u.nombre} {u.apellido}" if u else "—",
            "cam_id": log.cam_id,
            "action": log.action,
            "timestamp": log.timestamp,
        })
    return out

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




CAMERAS: Dict[str, CameraConfig] = {}

@app.post("/cameras")
def add_camera(cfg: CameraConfig, db: Session = Depends(get_db)):
    CAMERAS[cfg.cam_id] = cfg
    db.add(CameraAction(user_id=1, cam_id=cfg.cam_id, action="add"))
    db.commit()
    return {"ok": True}

@app.delete("/cameras/{cam_id}")
def delete_camera(cam_id: str, db: Session = Depends(get_db)):
    if cam_id in CAMERAS:
        CAMERAS.pop(cam_id)
        db.add(CameraAction(user_id=1, cam_id=cam_id, action="delete"))
        db.commit()
    return {"ok": True}

@app.get("/cameras")
def list_cameras():
    return list(CAMERAS.values())



class CameraWorker:
    def __init__(self, cam_id, src):
        self.cam_id = cam_id
        self.src = src
        self.clients: Set[WebSocket] = set()
        self.running = False
        self.task = None

        self.win_feats = deque(maxlen=SEQ_LEN)
        self.video_scores = deque(maxlen=VIDEO_MAX_SCORES)

        self.on_state = False
        self.W = None
        self.H = None

    async def start(self):
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self.run())

    async def run(self):
        import cv2

        cap = cv2.VideoCapture(int(self.src)) if self.src.isdigit() else cv2.VideoCapture(self.src)
        if not cap.isOpened():
            await self.broadcast({"type":"error"})
            return

        self.W, self.H = int(cap.get(3)), int(cap.get(4))

        while self.running:
            ok, frame = cap.read()
            if not ok:
                await self.broadcast({"type":"error"})
                break

            res = POSE.predict(frame, imgsz=IMGSZ, conf=CONF_POSE, verbose=False)[0]

            kps_f = None
            if res.keypoints is not None:
                xy = res.keypoints.xy.cpu().numpy()
                c  = res.keypoints.conf.cpu().numpy()

                order = np.argsort(-c.mean(axis=1))
                xy, c = xy[order[:TOPK]], c[order[:TOPK]]

                kps_f = np.concatenate([xy, c[...,None]], axis=-1)

            feat51 = pool_frame_to_51(kps_f, self.W, self.H)
            self.win_feats.append(feat51)

            p_win, p_vid = 0, 0

            if len(self.win_feats) == SEQ_LEN:
                X = np.stack(self.win_feats)
                X = (X - MU) / (SD + 1e-6)

                p_win = float(KERAS.predict(X[np.newaxis], verbose=0).ravel()[0])
                p_win = np.clip(p_win, 0, 1)

                while len(self.video_scores) > 60:
                    self.video_scores.popleft()

                self.video_scores.append(p_win)

                p_vid = float(np.mean(self.video_scores))


            now = time.time()

            if not self.on_state and p_vid >= THR_ON:
                self.on_state = True

                await self.broadcast({
                    "type":"alert",
                    "cam_id": self.cam_id,
                    "prob": p_vid,
                    "ts": now
                })


            elif self.on_state and p_vid <= THR_OFF:
                self.on_state = False

            try:
                if DRAW_OVERLAY:
                    shown = res.plot()
                else:
                    shown = frame
            except:
                shown = frame

            _, buf = cv2.imencode(".jpg", shown)
            jpg = base64.b64encode(buf).decode()

            await self.broadcast({
                "type":"frame",
                "cam_id": self.cam_id,
                "p_win": p_win,
                "p_vid": p_vid,
                "on": self.on_state,
                "ts": now,
                "jpg_b64": jpg,
            })

            await asyncio.sleep(0.005)

        cap.release()

    async def broadcast(self, msg):
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except:
                try: ws.close()
                except: pass
                self.clients.discard(ws)


@app.get("/admin/logs/alerts")
def admin_alert_logs(db: Session = Depends(get_db)):
    logs = db.query(AlertLog).all()
    return [{
        "cam_id": l.cam_id,
        "prob": l.prob,
        "timestamp": l.timestamp
    } for l in logs]
@app.delete("/admin/logs/alerts/clear")
def clear_alert_logs(db: Session = Depends(get_db)):
    db.query(AlertLog).delete()
    db.commit()
    return {"ok": True, "msg": "Alertas eliminadas"}

WORKERS: Dict[str, CameraWorker] = {}

@app.websocket("/ws/stream/{cam_id}")
async def ws_stream(ws: WebSocket, cam_id: str):
    await ws.accept()

    if cam_id not in CAMERAS:
        await ws.send_json({"type":"error","msg":"cam-not-found"})
        await ws.close()
        return

    worker = WORKERS.get(cam_id)
    if not worker:
        worker = CameraWorker(cam_id, CAMERAS[cam_id].src)
        WORKERS[cam_id] = worker
        await worker.start()

    worker.clients.add(ws)

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        worker.clients.discard(ws)



@app.post("/alerts/save")
def save_alert(decision: AlertDecision, db: Session = Depends(get_db)):
    if decision.accepted:
        db.add(AlertLog(
            cam_id=decision.cam_id,
            prob=decision.prob,
            timestamp=decision.timestamp
        ))
        db.commit()
        return {"ok": True, "msg": "Alerta guardada"}
    else:
        return {"ok": True, "msg": "Alerta descartada"}
