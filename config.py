import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Security and Authentication
    SECRET_KEY = os.getenv('SECRET_KEY', '3f8a7b6e5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7')
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///finance_chatbot.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Gemini AI Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyANghA3lWeU1NQub62m06VsbDlVSLrNQfI')
    
    # Google OAuth Configuration
    GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
    
    # Google Cloud Services Configuration
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'argon-liberty-452213-f0-a592714170b5.json')
    
    # Additional Configurations
    FINANCIAL_DATA_DIR = os.getenv('FINANCIAL_DATA_DIR', 'financial_data')
    VECTOR_STORE_PATH = os.getenv('VECTOR_STORE_PATH', 'vectorstore')
    
    @classmethod
    def validate_config(cls):
        """Validate required configurations"""
        required_keys = ['GEMINI_API_KEY']
        for key in required_keys:
            if not getattr(cls, key):
                raise ValueError(f"Missing required configuration: {key}")

# Validate configurations when imported
Config.validate_config()