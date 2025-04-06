from sqlalchemy import Column, Integer, Text, DateTime, Boolean, ForeignKey, String
from sqlalchemy.orm import relationship
from datetime import datetime
from . import db
import uuid

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'))
    message = Column(Text)
    response = Column(Text)
    translated_message = Column(Text)  # Stores English translation of user message
    translated_response = Column(Text)  # Stores original English version of bot response
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_financial = Column(Boolean, default=False)
    conversation_id = Column(String(36), index=True)
    language = Column(String(5))  # User's selected language
    
    # Translation status fields
    input_translation_status = Column(String(20), default='not_needed')  # not_needed/success/failed
    response_translation_status = Column(String(20), default='not_needed')  # not_needed/success/failed
    translation_error = Column(Text)  # Stores error details if any
    
    # Relationship with User model
    user = relationship("User", back_populates="chats")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.conversation_id:
            self.conversation_id = str(uuid.uuid4())
        
        # Default statuses based on language
        if self.language == 'en':
            self.input_translation_status = 'not_needed'
            self.response_translation_status = 'not_needed'
        else:
            self.input_translation_status = kwargs.get('input_translation_status', 'pending')
            self.response_translation_status = kwargs.get('response_translation_status', 'pending')

    def to_dict(self):
        """Enhanced dictionary conversion with translation metadata"""
        return {
            'id': self.id,
            'message': self.message,
            'response': self.response,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'is_financial': self.is_financial,
            'translated_message': self.translated_message,
            'translated_response': self.translated_response,
            'conversation_id': self.conversation_id,
            'language': self.language,
            'translation_status': {
                'input': self.input_translation_status,
                'response': self.response_translation_status
            },
            'translation_error': self.translation_error
        }

    def __repr__(self):
        return (f"<ChatHistory(id={self.id}, conversation={self.conversation_id}, "
                f"lang={self.language}, input_status={self.input_translation_status}, "
                f"response_status={self.response_translation_status})>")
