# app/schemas.py  –  Pydantic schemas (request / response)
from pydantic import BaseModel
from typing import Optional


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    nombre: str
    apellido: str
    email: str
    password: str
    charge: str
    codigo: Optional[str] = None


class CameraConfig(BaseModel):
    cam_id: str
    src: str


class AlertDecision(BaseModel):
    cam_id: str
    prob: float
    timestamp: float
    accepted: bool
