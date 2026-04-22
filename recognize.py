import datetime
import os
import time
import cv2
import pandas as pd
import numpy as np
import sys

def recognize_attendence():
    print("🎥 Iniciando reconocimiento facial...")
    
    # Verificar que exista el modelo entrenado
    model_path = "TrainingImageLabel" + os.sep + "Trainner.yml"
    
    if not os.path.exists(model_path):
        print("❌ Primero debes entrenar el modelo (ve a la sección de Entrenamiento)")
        print("💡 Ve a la pestaña 'Train' y haz clic en 'Iniciar Entrenamiento'")
        return False
    
    # Verificar que exista el archivo de estudiantes
    student_file = "StudentDetails" + os.sep + "StudentDetails.csv"
    if not os.path.exists(student_file):
        print("❌ No hay estudiantes registrados")
        print("💡 Ve a la pestaña 'Capture' y registra algunos estudiantes")
        return False
    
    # Inicializar el reconocedor de manera segura
    recognizer = None
    try:
        # Intentar diferentes métodos según la versión de OpenCV
        if hasattr(cv2, 'face'):
            if hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
                recognizer = cv2.face.LBPHFaceRecognizer_create()
            elif hasattr(cv2.face, 'createLBPHFaceRecognizer'):
                recognizer = cv2.face.createLBPHFaceRecognizer()
        elif hasattr(cv2, 'face_LBPHFaceRecognizer_create'):
            recognizer = cv2.face_LBPHFaceRecognizer_create()
        
        if recognizer is None:
            print("❌ No se pudo crear el reconocedor facial")
            print("💡 Ejecuta: pip install opencv-contrib-python==4.8.1.78")
            return False
            
        recognizer.read(model_path)
        print("✅ Modelo OpenCV cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return False
    
    # Cargar el clasificador de rostros
    harcascadePath = "haarcascade_default.xml"
    
    # Verificar que exista el archivo haarcascade
    if not os.path.exists(harcascadePath):
        print("❌ No se encuentra el archivo haarcascade_default.xml")
        print("💡 Asegúrate de tener este archivo en la carpeta del proyecto")
        return False
    
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    if faceCascade.empty():
        print("❌ Error al cargar el clasificador de rostros")
        return False
    
    # Leer estudiantes
    try:
        df = pd.read_csv(student_file)
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.lower()
        print(f"✅ {len(df)} estudiantes cargados")
    except Exception as e:
        print(f"❌ Error al leer estudiantes: {e}")
        return False
    
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    
    # Iniciar cámara
    print("📷 Abriendo cámara...")
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ No se pudo acceder a la cámara")
        print("💡 Verifica que tu cámara esté conectada y no esté siendo usada por otra aplicación")
        return False
        
    cam.set(3, 640)
    cam.set(4, 480)
    
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    
    print("✅ Cámara iniciada. Presiona 'q' para detener y guardar asistencia")
    print("=" * 50)
    
    while True:
        ret, im = cam.read()
        if not ret:
            print("Error al leer de la cámara")
            break
            
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5,
                minSize=(int(minW), int(minH)), flags=cv2.CASCADE_SCALE_IMAGE)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
            
            try:
                Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                
                if conf < 100:
                    # Buscar el nombre del estudiante
                    student = df[df['id'] == Id]
                    if len(student) > 0:
                        name = student['name'].values[0]
                    else:
                        name = "Unknown"
                    
                    tt = f"{Id}-{name}"
                    confidence = 100 - conf
                    
                    if confidence > 67:  # Confianza alta
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        
                        # Evitar duplicados en la misma sesión
                        if not ((attendance['Id'] == Id).any()):
                            attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
                            print(f"✅ Asistencia registrada: {name} (ID: {Id}) - Confianza: {confidence:.1f}%")
                        
                        cv2.putText(im, f"{tt} [✓]", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(im, f"{confidence:.1f}%", (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        cv2.putText(im, f"{tt} [?]", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(im, "Unknown", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            except Exception as e:
                cv2.putText(im, "Error", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Mostrar información en pantalla
        cv2.putText(im, f"Registros: {len(attendance)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(im, "Presiona 'q' para salir", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Reconocimiento Facial - Asistencia', im)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Guardar asistencia
    if len(attendance) > 0:
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour, Minute, Second = timeStamp.split(":")
        
        if not os.path.exists("Attendance"):
            os.makedirs("Attendance")
            
        fileName = "Attendance"+os.sep+"Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
        attendance.to_csv(fileName, index=False)
        print("=" * 50)
        print(f"✅ Asistencia guardada en: {fileName}")
        print(f"📊 Total de asistencias registradas: {len(attendance)}")
        print("=" * 50)
    else:
        print("⚠️ No se registraron asistencias")
    
    cam.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    recognize_attendence()