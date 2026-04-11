# uvicorn app.server:app --host 0.0.0.0 --port 8000
#
# Backend ligero (Render/Supabase):
#   - Auth, usuarios, cámaras (metadatos), alertas, admin
#   - Sirve el frontend estático
#   - NO corre inferencia (eso vive en app/server_inference.py, desplegado en RunPod)
#
# El navegador abre el WebSocket directamente contra la URL de RunPod,
# configurada vía INFERENCE_BASE_URL y expuesta al frontend en /config.js.

import os
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import Base, engine, SessionLocal
from app.models import User, AccessLog, CameraAction, AlertLog, Camera
from app.schemas import (
    LoginRequest, RegisterRequest, CameraConfig, AlertDecision,
)
from app.utils import verify_password
from app.crud import (
    create_user, get_user_by_email, add_access_log,
    add_camera_action, add_alert_log,
)

load_dotenv()
log = logging.getLogger("server")

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
ROOT_DIR   = Path(__file__).resolve().parent
STATIC_DIR = (ROOT_DIR / "static").resolve()

BOSS_CODE = os.getenv("BOSS_CODE", "20261")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# URL pública del servidor de inferencia (RunPod). Ej: https://abc-8000.proxy.runpod.net
INFERENCE_BASE_URL = os.getenv("INFERENCE_BASE_URL", "").rstrip("/")

LOG_RETENTION_DAYS = 7

FAILED_ATTEMPTS: dict[str, dict] = {}
MAX_ATTEMPTS       = 3
BLOCK_TIME_MINUTES = 5


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
# APP
# ==========================================================
app = FastAPI(title="Sistema de Videovigilancia - Backend")

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

@app.get("/login_fail")
def login_fail_page():
    return FileResponse(STATIC_DIR / "login_fail.html")


# ── Configuración inyectada al frontend ──────────────────
@app.get("/config.js")
def frontend_config():
    """
    Expone al navegador:
      - API_BASE_URL       -> este backend (Render, auth + DB + static)
      - INFERENCE_BASE_URL -> servidor de inferencia (RunPod, /ws, /overlay)
    """
    cfg = {
        "API_BASE_URL": "",
        "INFERENCE_BASE_URL": INFERENCE_BASE_URL,
    }
    js = f"""
window.SICHER_CONFIG = {json.dumps(cfg)};

window.sicherApiUrl = function(path) {{
  const base = (window.SICHER_CONFIG && window.SICHER_CONFIG.API_BASE_URL) || "";
  return base ? base.replace(/\\/$/, "") + path : path;
}};

window.sicherInferenceUrl = function(path) {{
  const base = (window.SICHER_CONFIG && window.SICHER_CONFIG.INFERENCE_BASE_URL) || "";
  return base ? base.replace(/\\/$/, "") + path : path;
}};

window.sicherWsUrl = function(path) {{
  const base = (window.SICHER_CONFIG && window.SICHER_CONFIG.INFERENCE_BASE_URL) || "";
  if (!base) {{
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return `${{proto}}://${{window.location.host}}${{path}}`;
  }}
  const url = new URL(base);
  const proto = url.protocol === "https:" ? "wss:" : "ws:";
  return `${{proto}}//${{url.host}}${{path}}`;
}};
""".strip()
    return Response(content=js, media_type="application/javascript")


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
# CÁMARAS  –  persistidas en DB (Supabase)
# ==========================================================
@app.get("/cameras")
def list_cameras(db: Session = Depends(get_db)):
    rows = db.query(Camera).all()
    return [{"cam_id": r.cam_id, "src": r.src} for r in rows]


@app.post("/cameras")
def add_camera(cfg: CameraConfig, db: Session = Depends(get_db)):
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
    db.query(Camera).filter(Camera.cam_id == cam_id).delete()
    db.commit()
    add_camera_action(db, 1, cam_id, "delete")
    return {"ok": True}


# ── Alertas (aceptar/rechazar) ────────────────────────────
@app.post("/alerts/save")
def save_alert(decision: AlertDecision, db: Session = Depends(get_db)):
    if decision.accepted:
        add_alert_log(db, decision.cam_id, decision.prob)
        return {"ok": True, "msg": "Alerta guardada"}
    return {"ok": True, "msg": "Alerta descartada"}
