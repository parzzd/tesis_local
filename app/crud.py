# app/crud.py  –  Operaciones CRUD sobre la base de datos
from sqlalchemy.orm import Session
from app.models import User, AccessLog, CameraAction, AlertLog
from app.utils import make_salt, hash_password


# ── Usuarios ──────────────────────────────────────────────
def create_user(db: Session, nombre: str, apellido: str, email: str,
                password: str, charge: str) -> User:
    salt = make_salt()
    user = User(
        nombre=nombre,
        apellido=apellido,
        username=email,
        password=hash_password(password, salt),
        salt=salt,
        charge=charge,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.username == email).first()


# ── Logs ──────────────────────────────────────────────────
def add_access_log(db: Session, user_id: int):
    db.add(AccessLog(user_id=user_id))
    db.commit()


def add_camera_action(db: Session, user_id: int, cam_id: str, action: str):
    db.add(CameraAction(user_id=user_id, cam_id=cam_id, action=action))
    db.commit()


def add_alert_log(db: Session, cam_id: str, prob: float):
    db.add(AlertLog(cam_id=cam_id, prob=prob))
    db.commit()
