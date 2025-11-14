from sqlalchemy import Column, Integer, String
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id        = Column(Integer, primary_key=True, index=True)
    nombre    = Column(String)
    apellido  = Column(String)
    username  = Column(String, unique=True, index=True)   # correo
    password  = Column(String)                             # hash
    charge    = Column(String)                             # operador / jefe
