# Real-Time Motion Detection System

A real-time motion detection system with parallel processing capabilities using MPI, featuring a web interface for monitoring and management.

## 🚀 Installation & Running Instructions

### For Mac Users

1. **Open Terminal and Navigate to Project Directory:**
   ```bash
   cd motiondetection/Real-Time Motion Detection
   ```

2. **Install MPI:**
   ```bash
   brew install open-mpi
   ```

3. **Create and Activate Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Open New Terminal Window:**
   ```bash
   cd motiondetection/Real-Time Motion Detection
   source venv/bin/activate
   mpirun -n 4 python app.py
   ```

6. **Access the Application:**
   - Open your web browser
   - Go to: http://localhost:3000
   - Login with:
     - Username: `user`
     - Password: `user`

### For Windows Users

1. **Open Command Prompt and Navigate to Project Directory:**
   ```cmd
   cd motiondetection\Real-Time Motion Detection
   ```

2. **Install MPI:**
   - Download MS-MPI from: https://www.microsoft.com/en-us/download/details.aspx?id=100593
   - Run the installer
   - Restart your computer

3. **Create and Activate Virtual Environment:**
   ```cmd
   python -m venv venv
   .\venv\Scripts\activate
   ```

4. **Install Dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

5. **Open New Command Prompt Window:**
   ```cmd
   cd motiondetection\Real-Time Motion Detection
   .\venv\Scripts\activate
   mpirun -n 4 python app.py
   ```

6. **Access the Application:**
   - Open your web browser
   - Go to: http://localhost:3000
   - Login with:
     - Username: `user`
     - Password: `user`

## 📸 Features
- Real-time motion detection
- Automatic recording on motion detection
- Timestamp-based video labeling
- Web-based monitoring interface
- User management system
- Admin panel for system control
- Parallel processing with MPI
- Motion detection logs
- Video playback with date/time information

## 📁 Project Structure
```
Real-Time Motion Detection/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── templates/            # HTML templates
│   ├── base.html        # Base template
│   ├── login.html       # Login page
│   ├── register.html    # Registration page
│   ├── dashboard.html   # Main dashboard
│   ├── admin.html       # Admin panel
│   └── play_recording.html  # Video playback
├── recordings/          # Recorded videos
└── logs/               # Motion logs
```

## 📝 License
MIT 