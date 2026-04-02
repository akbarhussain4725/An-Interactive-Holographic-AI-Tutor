from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
import numpy as np
import json
import pickle
import nltk
import random
import io
import os

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'edubot-secret-2024-xK9pQ'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///edubot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

lemmatizer = WordNetLemmatizer()

try:
    from tensorflow.keras.models import load_model
    model = load_model('chatbot_model.h5')
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    with open('intents.json', encoding='utf-8') as f:
        intents = json.load(f)
    MODEL_LOADED = True
except Exception as e:
    print(f"Model load error: {e}")
    MODEL_LOADED = False


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    grade = db.Column(db.String(20))
    school = db.Column(db.String(150))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('ChatMessage', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sender = db.Column(db.String(10), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    session_id = db.Column(db.String(50))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def clean_up_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]


def bag_of_words(sentence):
    token_words = clean_up_sentence(sentence)
    bag = [1 if w in token_words else 0 for w in words]
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure about that. Try asking about Maths, Physics, Chemistry, Biology, or Computer Science!"
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure about that. Try asking about a subject!"


@app.route('/')
def index():
    return render_template('landing.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    if request.method == 'POST':
        data = request.get_json()
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'success': False, 'message': 'Email already registered.'})
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'success': False, 'message': 'Username already taken.'})
        user = User(
            full_name=data['full_name'],
            email=data['email'],
            username=data['username'],
            grade=data.get('grade', ''),
            school=data.get('school', '')
        )
        user.set_password(data['password'])
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return jsonify({'success': True, 'redirect': url_for('chat')})
    return render_template('auth.html', mode='register')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    if request.method == 'POST':
        data = request.get_json()
        user = User.query.filter_by(username=data['username']).first()
        if user and user.check_password(data['password']):
            login_user(user)
            return jsonify({'success': True, 'redirect': url_for('chat')})
        return jsonify({'success': False, 'message': 'Invalid username or password.'})
    return render_template('auth.html', mode='login')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html', user=current_user)


@app.route('/api/message', methods=['POST'])
@login_required
def api_message():
    data = request.get_json()
    user_msg = data.get('message', '').strip()
    sess_id = data.get('session_id', 'default')

    if not user_msg:
        return jsonify({'error': 'Empty message'}), 400

    user_entry = ChatMessage(
        user_id=current_user.id,
        sender='user',
        message=user_msg,
        session_id=sess_id
    )
    db.session.add(user_entry)

    if MODEL_LOADED:
        intents_list = predict_class(user_msg)
        bot_reply = get_response(intents_list, intents)
    else:
        bot_reply = "Model not loaded. Please ensure chatbot_model.h5 is present."

    bot_entry = ChatMessage(
        user_id=current_user.id,
        sender='bot',
        message=bot_reply,
        session_id=sess_id
    )
    db.session.add(bot_entry)
    db.session.commit()

    return jsonify({
        'reply': bot_reply,
        'timestamp': datetime.utcnow().strftime('%I:%M %p')
    })


@app.route('/api/history')
@login_required
def api_history():
    messages = ChatMessage.query.filter_by(user_id=current_user.id)\
        .order_by(ChatMessage.timestamp.asc()).all()
    return jsonify([{
        'sender': m.sender,
        'message': m.message,
        'timestamp': m.timestamp.strftime('%b %d, %I:%M %p'),
        'session_id': m.session_id
    } for m in messages])


@app.route('/api/clear_history', methods=['POST'])
@login_required
def clear_history():
    ChatMessage.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({'success': True})


@app.route('/api/export_pdf')
@login_required
def export_pdf():
    messages = ChatMessage.query.filter_by(user_id=current_user.id)\
        .order_by(ChatMessage.timestamp.asc()).all()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=22, textColor=colors.HexColor('#1a1a2e'),
                                  spaceAfter=6)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                     fontSize=11, textColor=colors.HexColor('#6b7280'),
                                     spaceAfter=20)
    user_style = ParagraphStyle('User', parent=styles['Normal'],
                                 fontSize=10, textColor=colors.HexColor('#1d4ed8'),
                                 backColor=colors.HexColor('#eff6ff'),
                                 borderPadding=(8, 10, 8, 10),
                                 spaceAfter=6, leftIndent=60)
    bot_style = ParagraphStyle('Bot', parent=styles['Normal'],
                                fontSize=10, textColor=colors.HexColor('#1a1a2e'),
                                backColor=colors.HexColor('#f8fafc'),
                                borderPadding=(8, 10, 8, 10),
                                spaceAfter=6, rightIndent=60)
    label_style = ParagraphStyle('Label', parent=styles['Normal'],
                                  fontSize=8, textColor=colors.HexColor('#9ca3af'),
                                  spaceAfter=2)

    story = []
    story.append(Paragraph("EduBot Chat History", title_style))
    story.append(Paragraph(
        f"Student: {current_user.full_name}  |  Exported: {datetime.utcnow().strftime('%B %d, %Y %I:%M %p')}",
        subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor('#e5e7eb'), spaceAfter=16))

    for m in messages:
        ts = m.timestamp.strftime('%b %d, %I:%M %p')
        if m.sender == 'user':
            story.append(Paragraph(f"You  •  {ts}", label_style))
            story.append(Paragraph(m.message, user_style))
        else:
            story.append(Paragraph(f"EduBot  •  {ts}", label_style))
            story.append(Paragraph(m.message, bot_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True,
                     download_name=f'edubot_chat_{current_user.username}.pdf',
                     mimetype='application/pdf')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False, port=3000)
