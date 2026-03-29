from app.database import Base
from sqlalchemy import Column, Integer, String, Float, ForeignKey
import time
from pydantic import BaseModel
from typing import Optional

class User(Base):
    __tablename__ = "users"

    id        = Column(Integer, primary_key=True, index=True)
    nombre    = Column(String)
    apellido  = Column(String)
    username  = Column(String, unique=True, index=True)   # correo
    password  = Column(String)                             # hash
    charge    = Column(String)                             # operador / jefe
class AlertLog(Base):
    __tablename__ = "alert_logs"
    id = Column(Integer, primary_key=True)
    cam_id = Column(String)
    prob = Column(Float)
    timestamp = Column(Float, default=lambda: time.time())
class AccessLog(Base):
    __tablename__ = "access_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(Float, default=lambda: time.time())

class CameraAction(Base):
    __tablename__ = "camera_actions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    cam_id = Column(String)
    action = Column(String)
    timestamp = Column(Float, default=lambda: time.time())
class RegisterRequest(BaseModel):
    nombre: str
    apellido: str
    email: str
    password: str
    charge: str
    codigo: Optional[str] = None
class LoginRequest(BaseModel):
    email: str
    password: str
class CameraConfig(BaseModel):
    cam_id: str
    src: str
class AlertDecision(BaseModel):
    cam_id: str
    prob: float
    timestamp: float
    accepted: bool