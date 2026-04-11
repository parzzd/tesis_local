# app/models.py  –  SQLAlchemy ORM models
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Index
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from app.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


# ── Usuarios ──────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id       = Column(Integer, primary_key=True, index=True)
    nombre   = Column(String, nullable=False)
    apellido = Column(String, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)  # correo
    password = Column(String, nullable=False)                           # hash
    salt     = Column(String, nullable=False, default="")               # salt para PBKDF2
    charge   = Column(String, nullable=False, default="operador")       # operador | jefe

    access_logs    = relationship("AccessLog", back_populates="user", cascade="all, delete-orphan")
    camera_actions = relationship("CameraAction", back_populates="user", cascade="all, delete-orphan")


# ── Alertas ───────────────────────────────────────────────
class AlertLog(Base):
    __tablename__ = "alert_logs"

    id        = Column(Integer, primary_key=True)
    cam_id    = Column(String, nullable=False, index=True)
    prob      = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_alert_ts", "timestamp"),
    )


# ── Accesos ───────────────────────────────────────────────
class AccessLog(Base):
    __tablename__ = "access_logs"

    id        = Column(Integer, primary_key=True)
    user_id   = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=_utcnow, nullable=False)

    user = relationship("User", back_populates="access_logs")

    __table_args__ = (
        Index("ix_access_ts", "timestamp"),
    )


# ── Acciones sobre cámaras ────────────────────────────────
class CameraAction(Base):
    __tablename__ = "camera_actions"

    id        = Column(Integer, primary_key=True)
    user_id   = Column(Integer, ForeignKey("users.id"), nullable=False)
    cam_id    = Column(String, nullable=False)
    action    = Column(String, nullable=False)
    timestamp = Column(DateTime, default=_utcnow, nullable=False)

    user = relationship("User", back_populates="camera_actions")

    __table_args__ = (
        Index("ix_camaction_ts", "timestamp"),
    )


# ── Cámaras registradas ──────────────────────────────────
class Camera(Base):
    __tablename__ = "cameras"

    id     = Column(Integer, primary_key=True)
    cam_id = Column(String, unique=True, nullable=False, index=True)
    src    = Column(String, nullable=False)
