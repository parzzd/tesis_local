# app/crud.py
from sqlalchemy.orm import Session
from app import models

def create_user(db: Session, username: str, password_hash: str, camera_id: int):
    user = models.User(username=username, password=password_hash, camera_id=camera_id)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_user(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()
