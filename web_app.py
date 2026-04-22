from flask import Flask, render_template, request, jsonify, send_file
import os
import pandas as pd
from datetime import datetime
import shutil
import csv
import sys
from threading import Thread
import time

# Intentar importar cv2 con manejo de errores
try:
    import cv2
    CV2_AVAILABLE = True
    print("✅ OpenCV cargado correctamente")
except ImportError as e:
    CV2_AVAILABLE = False
    print(f"⚠️ OpenCV no disponible: {e}")
    print("💡 Ejecuta: pip install opencv-python==4.8.0.74")

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_cambia_esto'

# Importar tus módulos
try:
    import capture_image
    import train_image
    import recognize
    MODULES_AVAILABLE = True
except Exception as e:
    MODULES_AVAILABLE = False
    print(f"⚠️ Error importando módulos: {e}")

# Asegurar que las carpetas existan
folders = ['StudentDetails', 'TrainingImage', 'TrainingImageLabel', 'Attendance', 'templates', 'static/css', 'static/js']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Verificar que StudentDetails.csv exista
student_details_file = 'StudentDetails' + os.sep + 'StudentDetails.csv'
if not os.path.exists(student_details_file):
    with open(student_details_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Name'])

# Variables para controlar procesos
training_in_progress = False
recognition_in_progress = False

# ==================== RUTAS PRINCIPALES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture_page():
    return render_template('capture.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

@app.route('/recognize')
def recognize_page():
    return render_template('recognize.html')

@app.route('/attendance')
def attendance_page():
    attendance_files = []
    if os.path.exists('Attendance'):
        attendance_files = sorted([f for f in os.listdir('Attendance') if f.endswith('.csv')], reverse=True)
    return render_template('attendance.html', files=attendance_files)

@app.route('/reset')
def reset_page():
    return render_template('reset.html')

# ==================== API ENDPOINTS ====================
@app.route('/api/capture', methods=['POST'])
def api_capture():
    try:
        if not MODULES_AVAILABLE:
            return jsonify({'success': False, 'message': 'Módulos no disponibles. Revisa la instalación.'}), 500
            
        data = request.json
        student_id = data.get('id')
        name = data.get('name')
        
        def capture_thread():
            try:
                capture_image.takeImages(Id=student_id, name=name)
            except Exception as e:
                print(f"Error en captura: {e}")
        
        thread = Thread(target=capture_thread)
        thread.start()
        
        return jsonify({'success': True, 'message': f'Iniciando captura para {name}. La ventana de la cámara se abrirá.'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def api_train():
    global training_in_progress
    try:
        if not MODULES_AVAILABLE:
            return jsonify({'success': False, 'message': 'Módulos no disponibles. Revisa la instalación.'}), 500
            
        if training_in_progress:
            return jsonify({'success': False, 'message': 'Ya hay un entrenamiento en proceso'}), 400
        
        training_in_progress = True
        
        def train_thread():
            global training_in_progress
            try:
                train_image.TrainImages()
            except Exception as e:
                print(f"Error en entrenamiento: {e}")
            finally:
                training_in_progress = False
        
        thread = Thread(target=train_thread)
        thread.start()
        
        return jsonify({'success': True, 'message': 'Entrenamiento iniciado. Esto puede tomar varios minutos.'})
    except Exception as e:
        training_in_progress = False
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    global recognition_in_progress
    try:
        if not MODULES_AVAILABLE:
            return jsonify({'success': False, 'message': 'Módulos no disponibles. Revisa la instalación.'}), 500
            
        if recognition_in_progress:
            return jsonify({'success': False, 'message': 'Ya hay un reconocimiento en proceso'}), 400
        
        recognition_in_progress = True
        
        def recognize_thread():
            global recognition_in_progress
            try:
                recognize.recognize_attendence()
            except Exception as e:
                print(f"Error en reconocimiento: {e}")
            finally:
                recognition_in_progress = False
        
        thread = Thread(target=recognize_thread)
        thread.start()
        
        return jsonify({'success': True, 'message': 'Reconocimiento iniciado. La ventana de la cámara se abrirá.'})
    except Exception as e:
        recognition_in_progress = False
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/send-email', methods=['POST'])
def api_send_email():
    try:
        data = request.json
        receiver = data.get('email')
        
        attendance_files = [f for f in os.listdir('Attendance') if f.endswith('.csv')]
        if attendance_files:
            latest_file = max(attendance_files, key=lambda x: os.path.getctime(os.path.join('Attendance', x)))
            return jsonify({'success': True, 'message': f'Email preparado para {receiver} con archivo {latest_file}'})
        else:
            return jsonify({'success': False, 'message': 'No hay archivos de asistencia'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def api_reset():
    try:
        folders = ["TrainingImage", "TrainingImageLabel", "Attendance"]
        for folder in folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)
        return jsonify({'success': True, 'message': 'Sistema reiniciado correctamente'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/attendance-files')
def get_attendance_files():
    files = []
    if os.path.exists('Attendance'):
        for file in os.listdir('Attendance'):
            if file.endswith('.csv'):
                file_path = os.path.join('Attendance', file)
                stat = os.stat(file_path)
                files.append({
                    'name': file,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
    return jsonify(sorted(files, key=lambda x: x['modified'], reverse=True))

@app.route('/api/attendance-view/<filename>')
def view_attendance(filename):
    try:
        file_path = os.path.join('Attendance', filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return jsonify(df.to_dict('records'))
        return jsonify({'error': 'Archivo no encontrado'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/student-count')
def student_count():
    try:
        student_file = 'StudentDetails' + os.sep + 'StudentDetails.csv'
        if os.path.exists(student_file):
            df = pd.read_csv(student_file)
            return jsonify({'count': len(df)})
        return jsonify({'count': 0})
    except Exception as e:
        return jsonify({'count': 0})

@app.route('/api/image-count')
def image_count():
    try:
        image_folder = 'TrainingImage'
        if os.path.exists(image_folder):
            images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
            return jsonify({'count': len(images)})
        return jsonify({'count': 0})
    except Exception as e:
        return jsonify({'count': 0})

@app.route('/api/training-status')
def training_status():
    global training_in_progress
    return jsonify({'training': training_in_progress})

@app.route('/api/recognition-status')
def recognition_status():
    global recognition_in_progress
    return jsonify({'recognizing': recognition_in_progress})

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join('Attendance', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "Archivo no encontrado", 404

# ==================== EJECUCIÓN ====================
if __name__ == '__main__':
    print("=" * 50)
    print("🚀 SISTEMA DE ASISTENCIA FACIAL WEB")
    print("=" * 50)
    print(f"✅ Flask iniciado")
    print(f"✅ OpenCV disponible: {CV2_AVAILABLE}")
    print(f"✅ Módulos disponibles: {MODULES_AVAILABLE}")
    print("🌐 Servidor en: http://127.0.0.1:5000")
    print("=" * 50)
    
    if not CV2_AVAILABLE or not MODULES_AVAILABLE:
        print("⚠️ ADVERTENCIA: Algunas dependencias no están instaladas correctamente")
        print("💡 Ejecuta el siguiente comando para instalar todo:")
        print("   pip install numpy==1.24.3 opencv-python==4.8.0.74 opencv-contrib-python==4.8.0.74 pandas==2.0.3 Flask==2.3.2 Pillow==10.0.0")
        print("=" * 50)
    
    app.run(debug=True, host='127.0.0.1', port=5000)