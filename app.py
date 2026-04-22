from flask import Flask, render_template, request, jsonify, send_file
import os
import pandas as pd
from datetime import datetime
import shutil
import csv
import subprocess
import sys

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_cambia_esto'

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
        data = request.json
        student_id = data.get('id')
        name = data.get('name')
        
        # Ejecutar capture_image.py con parámetros
        # Nota: Debes modificar capture_image.py para aceptar argumentos
        result = subprocess.run([
            sys.executable, 
            'capture_image.py',
            '--id', str(student_id),
            '--name', name
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({'success': True, 'message': f'Capturas guardadas para {name}'})
        else:
            return jsonify({'success': False, 'message': result.stderr}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    
@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        # Ejecutar train_image.py
        result = subprocess.run([
            sys.executable, 
            'train_image.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'Modelo entrenado exitosamente'})
        else:
            return jsonify({'success': False, 'message': result.stderr}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    
@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    try:
        # Ejecutar recognize.py
        result = subprocess.run([
            sys.executable, 
            'recognize.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'Reconocimiento completado'})
        else:
            return jsonify({'success': False, 'message': result.stderr}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
@app.route('/api/send-email', methods=['POST'])
def api_send_email():
    try:
        data = request.json
        receiver = data.get('email')
        return jsonify({'success': True, 'message': f'Email enviado a {receiver}'})
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
    """Contar estudiantes registrados"""
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
    """Contar imágenes en la carpeta TrainingImage"""
    try:
        image_folder = 'TrainingImage'
        if os.path.exists(image_folder):
            images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
            return jsonify({'count': len(images)})
        return jsonify({'count': 0})
    except Exception as e:
        return jsonify({'count': 0, 'error': str(e)})

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join('Attendance', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "Archivo no encontrado", 404

# ==================== EJECUCIÓN ====================
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)