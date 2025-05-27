from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
import cv2
import numpy as np
from threading import Thread
import time
import shutil
import tempfile
from mpi4py import MPI

# Geçici dizin oluştur
temp_dir = tempfile.mkdtemp()

# MPI başlatma
try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # MPI paylaşımlı bellek ayarları
    os.environ['OMPI_MCA_btl'] = '^sm'
    os.environ['OMPI_MCA_btl_tcp_if_include'] = 'lo'
except Exception as e:
    print(f"MPI başlatma hatası: {e}")
    comm = None
    rank = 0
    size = 1

app = Flask(__name__)
app.config['SECRET_KEY'] = 'password'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///security.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Global variables for video streaming
camera = None
motion_detector = None
is_recording = False
last_motion_time = None
motion_status = {'motion': False, 'last_motion': None}

class MotionDetector:
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        self.recording = False
        self.output_video = None
        self.motion_detected = False
        self.last_motion_time = 0
        self.alert_cooldown = 5
        self.motion_start_time = None
        self.current_recording_filename = None
        self.frame_written = False
        
        # Sadece rank 0 için dizin oluşturma
        if rank == 0:
            if not os.path.exists('recordings'):
                os.makedirs('recordings')
            if not os.path.exists('logs'):
                os.makedirs('logs')
            self.log_file = 'logs/motion_log.txt'
            self.excel_file = 'logs/motion_log.xlsx'
            self.initialize_logs()
    
    def initialize_logs(self):
        if rank == 0:
            with open(self.log_file, 'w') as f:
                f.write("Motion Detection Log\n")
                f.write("===================\n\n")
    
    def log_motion_event(self, start_time=None, end_time=None):
        if rank == 0:
            timestamp = datetime.now()
            date_str = timestamp.strftime("%Y-%m-%d")
            time_str = timestamp.strftime("%H:%M:%S")
            
            with open(self.log_file, 'a') as f:
                if start_time:
                    f.write(f"Motion Detected - {date_str} {time_str}\n")
                if end_time:
                    duration_seconds = (end_time - start_time).total_seconds()
                    f.write(f"Motion Ended - Duration: {duration_seconds:.2f} seconds\n")
                    f.write("-" * 50 + "\n")
    
    def start_recording(self):
        if rank == 0 and not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"motion_{timestamp}.mp4"
            filepath = os.path.join('recordings', filename)
            # Video codec'ini değiştiriyoruz
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec'i
            fps = 20.0
            frame_size = (640, 480)
            self.output_video = cv2.VideoWriter(filepath, fourcc, fps, frame_size)
            if not self.output_video.isOpened():
                print("Video kayıt başlatılamadı!")
                return
            self.recording = True
            self.motion_start_time = datetime.now()
            self.log_motion_event(start_time=self.motion_start_time)
            global motion_status
            motion_status['motion'] = True
            motion_status['last_motion'] = self.motion_start_time.strftime('%Y-%m-%d %H:%M:%S')
            self.current_recording_filename = filename
            self.frame_written = False
            print(f"Kayıt başladı: {filename}")
    
    def stop_recording(self):
        if rank == 0 and self.recording:
            if self.output_video is not None:
                self.output_video.release()
            self.recording = False
            self.output_video = None
            self.log_motion_event(end_time=datetime.now(), start_time=self.motion_start_time)
            global motion_status
            motion_status['motion'] = False
            filepath = os.path.join('recordings', self.current_recording_filename)
            try:
                if not self.frame_written or os.path.getsize(filepath) < 100*1024:
                    os.remove(filepath)
                    print(f"Boş kayıt silindi: {self.current_recording_filename}")
                else:
                    print(f"Kayıt tamamlandı: {self.current_recording_filename}")
            except Exception as e:
                print(f"Kayıt dosyası işlenirken hata: {e}")
            self.current_recording_filename = None
            self.frame_written = False
    
    def process_frame(self, frame):
        fg_mask = self.background_subtractor.apply(frame)
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return frame, fg_mask, motion_detected

def generate_frames():
    global camera, motion_detector
    
    if rank == 0:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("Kamera açılamadı!")
                return
            # Kamera ayarlarını optimize et
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 20)
        if motion_detector is None:
            motion_detector = MotionDetector()
    
    while True:
        if rank == 0:
            success, frame = camera.read()
            if not success:
                print("Kameradan kare okunamadı!")
                break
            
            # Frame'i işle
            processed_frame, fg_mask, motion_detected = motion_detector.process_frame(frame)
            
            # Hareket algılandıysa kaydı başlat
            if motion_detected:
                if not motion_detector.motion_detected:
                    motion_detector.motion_detected = True
                    motion_detector.start_recording()
            else:
                if motion_detector.motion_detected:
                    motion_detector.motion_detected = False
                    motion_detector.stop_recording()
            
            # Durum bilgisini ekle
            status = "RECORDING" if motion_detector.recording else "Monitoring"
            cv2.putText(processed_frame, f'Status: {status}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Kayıt yap
            if motion_detector.recording and motion_detector.output_video is not None:
                try:
                    motion_detector.output_video.write(processed_frame)
                    motion_detector.frame_written = True
                except Exception as e:
                    print(f"Frame yazma hatası: {e}")
            
            # Görüntüyü encode et ve yayınla
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), default='user')  # 'admin' or 'user'
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Bu kullanıcı adı zaten kullanılıyor')
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first():
            flash('Bu e-posta adresi zaten kayıtlı')
            return redirect(url_for('register'))
            
        user = User(username=username, email=email, role='admin')  # Varsayılan rolü admin yapıyoruz
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Kayıt başarılı')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Get list of recordings and logs
    recordings = []
    logs = []
    
    # Read recordings directory
    if os.path.exists('recordings'):
        recordings = [f for f in os.listdir('recordings') if f.endswith('.mp4')]
    
    # Read logs
    if os.path.exists('logs/motion_log.txt'):
        with open('logs/motion_log.txt', 'r') as f:
            logs = f.readlines()
    
    return render_template('dashboard.html', 
                         recordings=recordings, 
                         logs=logs,
                         is_admin=current_user.role == 'admin')

@app.route('/admin')
@login_required
def admin():
    if current_user.role != 'admin':
        flash('Bu sayfaya erişim yetkiniz yok')
        return redirect(url_for('dashboard'))
    users = User.query.all()
    return render_template('admin.html', users=users)

@app.route('/update_role', methods=['POST'])
def update_role():
    user_id = request.form.get('user_id')
    new_role = request.form.get('role')
    user = User.query.get(user_id)
    if user:
        user.role = new_role
        db.session.commit()
    return redirect(url_for('admin'))

@app.route('/logout')
@login_required
def logout():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    logout_user()
    return redirect(url_for('login'))

@app.route('/recordings/<path:filename>')
@login_required
def play_recording(filename):
    # Dosya adından tarih ve saat bilgisini çıkar
    try:
        # motion_YYYYMMDD_HHMMSS.mp4 formatından tarih ve saat bilgisini al
        date_str = filename.split('_')[1]
        time_str = filename.split('_')[2].split('.')[0]
        
        # Tarih ve saat formatını düzenle
        recording_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        recording_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
    except:
        recording_date = "Bilinmiyor"
        recording_time = "Bilinmiyor"
    
    return render_template('play_recording.html', 
                         filename=filename,
                         recording_date=recording_date,
                         recording_time=recording_time)

@app.route('/recordings/raw/<path:filename>')
@login_required
def serve_recording(filename):
    return send_from_directory('recordings', filename, mimetype='video/mp4')

@app.route('/motion_status')
@login_required
def motion_status_api():
    return jsonify(motion_status)

@app.route('/delete_user', methods=['POST'])
def delete_user():
    user_id = request.form.get('user_id')
    user = User.query.get(user_id)
    if user:
        # Admin kullanıcısını silmeye çalışıyorsa engelle
        if user.username == 'admin':
            return jsonify({'success': False, 'error': 'Admin kullanıcısı silinemez'})
        db.session.delete(user)
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Kullanıcı bulunamadı'})

if __name__ == '__main__':
    try:
        if rank == 0:
            with app.app_context():
                db.create_all()
                admin_user = User.query.filter_by(username='admin').first()
                if admin_user:
                    if admin_user.role != 'admin':
                        admin_user.role = 'admin'
                        db.session.commit()
                else:
                    admin = User(username='admin', email='admin@example.com', role='admin')
                    admin.set_password('1')
                    db.session.add(admin)
                    db.session.commit()
            app.run(debug=False, host='0.0.0.0', port=3000)
    finally:
        # Geçici dizini temizle
        try:
            shutil.rmtree(temp_dir)
        except:
            pass 