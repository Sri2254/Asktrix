from flask_login import UserMixin
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from . import db

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    email = Column(String(100), unique=True)
    password = Column(String(100))
    name = Column(String(100))
    # In models/user.py
    chats = relationship("ChatHistory", back_populates="user", order_by="ChatHistory.timestamp.desc()")

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name}, email={self.email})>"
