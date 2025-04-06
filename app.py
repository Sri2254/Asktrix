import os
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash,session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db
from models.user import User
from models.chat import ChatHistory
from config import Config
import google.generativeai as genai
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from typing import List, Dict
from flask_dance.contrib.google import make_google_blueprint, google
from flask_dance.consumer import oauth_authorized, oauth_error
from flask_dance.consumer.storage.sqla import SQLAlchemyStorage
from sqlalchemy.orm.exc import NoResultFound
from gtts import gTTS
import pygame
from io import BytesIO
from translate import Translator as TranslateLibTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import time
import logging
from pydantic import BaseModel

# Initialize Flask application
project_dir = os.path.abspath(os.path.dirname(__file__))
templates_dir = os.path.join(project_dir, 'templates')
logging.basicConfig(level=logging.INFO)

if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)

app = Flask(__name__, template_folder=templates_dir)
app.config.from_object(Config)
app.secret_key = app.config['SECRET_KEY']

# Database and authentication setup
db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# Google OAuth configuration
google_bp = make_google_blueprint(
    client_id=app.config['GOOGLE_CLIENT_ID'],
    client_secret=app.config['GOOGLE_CLIENT_SECRET'],
    scope=["profile", "email"],
    storage=SQLAlchemyStorage(User, db.session, user=current_user)
)
app.register_blueprint(google_bp, url_prefix="/login")

# Application constants
DATA_DIR = "financial_data"
VECTOR_STORE_PATH = "vectorstore"
financial_keywords = ['stock', 'investment', 'portfolio', 'finance', 'money', 'save', 'spend', 'budget', 'debt']

LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'ta': 'Tamil',
    'fr': 'French',
    'es': 'Spanish'
}

class FallbackLLM:
    """Fallback language model when primary services are unavailable"""
    def generate_content(self, prompt):
        class Response:
            text = """Financial advice placeholder:
            For the question: {prompt}
            [System Notice: Our AI services are currently unavailable]
            Please try again later or consult a human financial advisor."""
        return Response()

# Configure Gemini AI with proper error handling
try:
    genai.configure(
        api_key='AIzaSyANghA3lWeU1NQub62m06VsbDlVSLrNQfI')
    model_names_to_try = ['gemini-1.5-pro', 'gemini-1.5-flash','gemini-pro', 'models/gemini-pro']
    model = None
    for model_name in model_names_to_try:
        try:
            print(f"Attempting to connect to {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Hello")
            print(f"Successfully connected to {model_name}: {response.text[:20]}...")
            break
        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")
    if not model:
        raise RuntimeError("Could not initialize any supported Gemini model")
except Exception as e:
    print(f"Critical error configuring Gemini: {str(e)}")
    class DummyModel:
        def generate_content(self, prompt):
            class Response:
                text = f"""As a financial advisor, I would respond to: "{prompt}"
                Key points:
                1. This is a simulated response (Gemini API not configured)
                2. For real financial advice, please configure the API
                3. Always consult with a human financial expert"""
            return Response()
    model = DummyModel()
    print("WARNING: Using dummy model - real API not properly configured")


class ChatUtilities:
    """Utility class for text-to-speech and translation"""
    @staticmethod
    def text_to_speech(text, lang='en'):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            temp_file = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4()}.mp3")
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(temp_file)
            time.sleep(0.5)
            
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            try:
                os.remove(temp_file)
            except:
                pass
        except Exception as e:
            print(f"Text-to-speech error: {str(e)}")

    @staticmethod
    def translate_text(text, dest_lang='en'):
        try:
            if len(text) > 500:
                text = text[:500] + "... [truncated]"
            translator = TranslateLibTranslator(to_lang=dest_lang)
            return translator.translate(text)
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text

class FinancialDataProcessor:
    """Handles document processing and vector storage for RAG"""
    def __init__(self):
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key='AIzaSyANghA3lWeU1NQub62m06VsbDlVSLrNQfI'
            )
        except Exception as e:
            print(f"Error initializing embeddings: {str(e)}")
            raise

    def load_documents(self) -> List[Dict]:
        documents = []
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR, exist_ok=True)
            return documents
            
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.csv'):
                filepath = os.path.join(DATA_DIR, filename)
                try:
                    loader = CSVLoader(filepath)
                    loaded_docs = loader.load()
                    documents.extend([{
                        "page_content": doc.page_content,
                        "metadata": {"source": filename}
                    } for doc in loaded_docs])
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        return documents

    def create_vectorstore(self, documents: List[Dict]):
        proper_docs = [Document(
            page_content=doc["page_content"],
            metadata=doc.get("metadata", {"source": "unknown"})
        ) for doc in documents]
        
        splitted_docs = self.text_splitter.split_documents(proper_docs)
        self.vectorstore = FAISS.from_documents(splitted_docs, self.embeddings)
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        self.vectorstore.save_local(VECTOR_STORE_PATH)

    def load_vectorstore(self):
        try:
            self.vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            self.vectorstore = FAISS.from_texts(
                ["Financial knowledge base is being initialized..."], 
                self.embeddings
            )
            self.vectorstore.save_local(VECTOR_STORE_PATH)

    def get_relevant_documents(self, query: str, k: int = 3) -> List[Dict]:
        if not self.vectorstore:
            self.load_vectorstore()
            
        docs = self.vectorstore.similarity_search(query, k=k)
        return [{
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown")
        } for doc in docs]

# In the ChatProcessor class:
class ChatProcessor:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.vectorstore = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

    def create_chain(self):
        retriever = self.vectorstore.as_retriever()
        
        # Try multiple model names
        model_names_to_try = ['gemini-1.5-pro', 'gemini-pro', 'models/gemini-pro']
        llm = None
        
        for model_name in model_names_to_try:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.2,
                    google_api_key='AIzaSyANghA3lWeU1NQub62m06VsbDlVSLrNQfI'
                )
                # Test the connection
                llm.invoke("Test connection")
                print(f"Successfully connected to {model_name}")
                break
            except Exception as e:
                print(f"Error with model {model_name}: {str(e)}")
        
        if not llm:
            raise RuntimeError("Could not initialize any supported Gemini model")
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True
        )

    def ask(self, query):
        try:
            result = self.chain.invoke({"question": query})
            return result.get("answer", "Sorry, I couldn't process that request.")
        except Exception as e:
            logging.error(f"Chain invocation error: {str(e)}")
            return "I'm having trouble accessing my knowledge base. Please try again later."
try:
    processor = FinancialDataProcessor()
    processor.load_vectorstore()
    chat_processor = ChatProcessor(processor.embeddings)
    chat_processor.vectorstore = processor.vectorstore
    chat_processor.create_chain()
except Exception as e:
    print(f"Initialization error: {str(e)}")
    processor = None
    chat_processor = None

# Create database tables
with app.app_context():
    db.create_all()

# Authentication routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/auth/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'
        
        user = db.session.execute(db.select(User).filter_by(email=email)).scalar_one_or_none()
        if user and check_password_hash(user.password, password):
            login_user(user, remember=remember)
            return redirect(url_for('chat'))
        
        flash('Invalid credentials', 'error')
    return render_template('auth/login.html')

@app.route('/auth/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')

        if User.query.filter_by(email=email).first():
            flash("Email already registered", 'error')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(email=email, name=name, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registered successfully. Please login.", 'success')
        return redirect(url_for('login'))
    return render_template('auth/register.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = db.session.execute(db.select(User).filter_by(email=email)).scalar_one_or_none()
        if user:
            flash('Password reset link sent to your email', 'info')
        else:
            flash('No account found with that email', 'error')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/auth/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# Chat history routes
@app.route('/history')
@login_required
def history():
    conversations = db.session.query(
        ChatHistory.conversation_id,
        db.func.max(ChatHistory.timestamp).label('last_updated'),
        db.func.count(ChatHistory.id).label('message_count')
    ).filter_by(
        user_id=current_user.id
    ).group_by(
        ChatHistory.conversation_id
    ).order_by(
        db.desc('last_updated')
    ).all()
    
    conversation_data = []
    for conv in conversations:
        messages = ChatHistory.query.filter_by(
            user_id=current_user.id,
            conversation_id=conv.conversation_id
        ).order_by(
            ChatHistory.timestamp.asc()
        ).all()
        
        is_financial = any(msg.is_financial for msg in messages)
        
        conversation_data.append({
            'id': conv.conversation_id,
            'timestamp': conv.last_updated,
            'message_count': conv.message_count,
            'is_financial': is_financial,
            'language': messages[0].language if messages else 'en',
            'messages': messages
        })
    
    return render_template('history.html', conversations=conversation_data)

@app.route('/api/conversations', methods=['GET'])
@login_required
def get_conversations():
    conversations = db.session.query(
        ChatHistory.conversation_id,
        db.func.max(ChatHistory.timestamp).label('max_timestamp'),
        db.func.count(ChatHistory.id).label('message_count')
    ).filter_by(
        user_id=current_user.id
    ).group_by(
        ChatHistory.conversation_id
    ).order_by(
        db.desc('max_timestamp')
    ).all()
    
    conversation_data = []
    for conv in conversations:
        first_message = ChatHistory.query.filter_by(
            user_id=current_user.id,
            conversation_id=conv.conversation_id
        ).order_by(
            ChatHistory.timestamp.asc()
        ).first()
        
        conversation_data.append({
            'id': conv.conversation_id,
            'title': first_message.message[:50] + ('...' if len(first_message.message) > 50 else ''),
            'timestamp': conv.max_timestamp.strftime('%Y-%m-%d %H:%M'),
            'message_count': conv.message_count,
            'language': first_message.language
        })
    
    return jsonify(conversation_data)

# Chat functionality routes
@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    if request.method == 'POST':
        user_input = request.form.get('message')
        language = request.form.get('language', 'en')
        
        translated_input = ChatUtilities.translate_text(user_input, 'en') if language != 'en' else user_input
        
        context_docs = processor.get_relevant_documents(translated_input) if processor else []
        context_text = "\n\n".join([doc["content"] for doc in context_docs])

        if chat_processor:
            english_reply = chat_processor.ask(translated_input)
        else:
            response = model.generate_content(translated_input)
            english_reply = response.text

        final_reply = ChatUtilities.translate_text(english_reply, language) if language != 'en' else english_reply
        
        chat = ChatHistory(
            user_id=current_user.id,
            message=user_input,
            response=final_reply,
            translated_message=translated_input,
            translated_response=english_reply,
            is_financial=bool(context_docs),
            language=language,
            conversation_id=str(uuid.uuid4())
        )
        db.session.add(chat)
        db.session.commit()

        return jsonify({'reply': final_reply})
    
    return render_template('chat.html')

@app.route('/api/chat', methods=["POST"])
@login_required
def api_chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Message is required"}), 400
            
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400
            
        language = data.get("language", "en")
        read_aloud = data.get("read_aloud", False)
        
        # Handle conversation ID
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            last_chat = ChatHistory.query.filter_by(
                user_id=current_user.id
            ).order_by(ChatHistory.timestamp.desc()).first()
            
            if last_chat and (datetime.utcnow() - last_chat.timestamp).total_seconds() < 3600:
                conversation_id = last_chat.conversation_id
            else:
                conversation_id = str(uuid.uuid4())

        # Translation
        translated_message = ChatUtilities.translate_text(user_message, 'en') if language != 'en' else user_message
        
        # Process query with fallback
        try:
            if chat_processor:
                english_reply = chat_processor.ask(translated_message)
            else:
                response = model.generate_content(translated_message)
                english_reply = response.text
        except Exception as e:
            logging.error(f"Response generation error: {str(e)}")
            english_reply = "I encountered an error processing your request. Please try again."

        # Translate response if needed
        final_reply = ChatUtilities.translate_text(english_reply, language) if language != 'en' else english_reply
        
        # Save to database
        chat = ChatHistory(
            user_id=current_user.id,
            message=user_message,
            response=final_reply,
            translated_message=translated_message,
            translated_response=english_reply,
            is_financial=any(keyword in user_message.lower() for keyword in financial_keywords),
            language=language,
            conversation_id=conversation_id
        )
        db.session.add(chat)
        db.session.commit()

        # Text-to-speech if requested
        if read_aloud and final_reply:
            try:
                ChatUtilities.text_to_speech(final_reply, lang=language)
            except Exception as e:
                logging.error(f"Text-to-speech error: {str(e)}")

        return jsonify({
            "reply": final_reply,
            "conversation_id": conversation_id,
            "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        return jsonify({
            "reply": "Our financial advisory service is currently unavailable. Please try again later.",
            "conversation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        }), 200
    
@app.route('/api/chat/read', methods=['POST'])
@login_required
def read_aloud():
    data = request.json
    text = data.get('text', '')
    lang = data.get('lang', 'en')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        ChatUtilities.text_to_speech(text, lang)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/new', methods=['POST'])
@login_required
def new_conversation():
    conversation_id = request.json.get('conversation_id', str(uuid.uuid4()))
    return jsonify({
        'status': 'success',
        'conversation_id': conversation_id
    })

@app.route('/api/chat/history', methods=['GET'])
@login_required
def chat_history():
    try:
        chats = db.session.execute(
            db.select(ChatHistory)
            .filter_by(user_id=current_user.id)
            .order_by(ChatHistory.timestamp.desc())
            .limit(10)
        ).scalars().all()
        
        return jsonify([{
            'message': chat.message,
            'response': chat.response,
            'timestamp': chat.timestamp.strftime('%Y-%m-%d %H:%M'),
            'is_financial': chat.is_financial,
            'translated_message': chat.translated_message,
            'translated_response': chat.translated_response
        } for chat in chats])
    except Exception as e:
        app.logger.error(f"Error fetching chat history: {str(e)}")
        return jsonify({"error": "Could not retrieve chat history"}), 500

# File upload route
@app.route('/upload', methods=['POST'])
@login_required
def upload_csv():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('chat'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('chat'))

    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(DATA_DIR, file.filename)
        file.save(filepath)
        flash('File uploaded. Rebuilding vector store...', 'info')
        
        documents = processor.load_documents()
        processor.create_vectorstore(documents)
        flash('Vector store rebuilt successfully.', 'success')
    else:
        flash('Invalid file format. Only CSV allowed.', 'error')
    return redirect(url_for('chat'))

# Additional API routes
@app.route('/api/languages', methods=['GET'])
def get_languages():
    return jsonify(LANGUAGES)

@app.route('/test/embeddings')
def test_embeddings():
    if not processor or not processor.embeddings:
        return jsonify({"status": "error", "message": "Embeddings not initialized"}), 500
    try:
        test_embedding = processor.embeddings.embed_query("test")
        return jsonify({"status": "success", "embedding_size": len(test_embedding)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Google OAuth handlers
@oauth_authorized.connect_via(google_bp)
def google_logged_in(blueprint, token):
    if not token:
        flash("Failed to log in with Google", category="error")
        return False
    
    resp = blueprint.session.get("/oauth2/v2/userinfo")
    if not resp.ok:
        flash("Failed to fetch user info from Google", category="error")
        return False
    
    google_info = resp.json()
    email = google_info["email"]
    
    try:
        user = User.query.filter_by(email=email).one()
    except NoResultFound:
        user = User(
            email=email,
            name=google_info.get("name", email.split('@')[0]),
            password=generate_password_hash(os.urandom(24).hex())
        )
        db.session.add(user)
        db.session.commit()
    
    login_user(user)
    flash("Successfully logged in with Google", category="success")
    return redirect(url_for('chat'))

@oauth_error.connect_via(google_bp)
def google_error(blueprint, error, error_description=None, error_uri=None):
    flash(f"Google OAuth error: {error}", category="error")
    return redirect(url_for('login'))

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/api/theme', methods=['POST'])
@login_required
def set_theme():
    theme = request.json.get('theme', 'light')
    # Store in user session (or database if you want persistence)
    session['theme'] = theme
    return jsonify({'status': 'success', 'theme': theme})

@app.route('/api/theme', methods=['GET'])
@login_required
def get_theme():
    return jsonify({'theme': session.get('theme', 'light')})

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    app.run(debug=True)