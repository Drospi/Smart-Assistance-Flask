import os
import cv2
import numpy as np
from PIL import Image
import sys

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    faces = []
    Ids = []
    
    print(f"📸 Procesando {len(imagePaths)} imágenes...")
    
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            # Extraer ID del nombre del archivo (formato: nombre.ID.numero.jpg)
            parts = os.path.split(imagePath)[-1].split(".")
            if len(parts) >= 2:
                Id = int(parts[1])
                faces.append(imageNp)
                Ids.append(Id)
            else:
                print(f"⚠️ Formato incorrecto: {imagePath}")
        except Exception as e:
            print(f"❌ Error procesando {imagePath}: {e}")
            continue
    
    print(f"✅ {len(faces)} imágenes válidas encontradas")
    return faces, Ids

def TrainImages():
    print("🚀 Iniciando entrenamiento del modelo...")
    print("=" * 50)
    
    # Crear carpeta si no existe
    if not os.path.exists("TrainingImageLabel"):
        os.makedirs("TrainingImageLabel")
    
    # Verificar que existan imágenes
    if not os.path.exists("TrainingImage") or len(os.listdir("TrainingImage")) == 0:
        print("❌ No hay imágenes en la carpeta TrainingImage")
        print("💡 Primero debes capturar rostros en la sección 'Capture'")
        return False
    
    # Inicializar el reconocedor
    recognizer = None
    try:
        # Probar diferentes métodos según la versión de OpenCV
        if hasattr(cv2, 'face'):
            if hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                print("✅ Usando cv2.face.LBPHFaceRecognizer_create()")
            elif hasattr(cv2.face, 'createLBPHFaceRecognizer'):
                recognizer = cv2.face.createLBPHFaceRecognizer()
                print("✅ Usando cv2.face.createLBPHFaceRecognizer()")
        elif hasattr(cv2, 'face_LBPHFaceRecognizer_create'):
            recognizer = cv2.face_LBPHFaceRecognizer_create()
            print("✅ Usando cv2.face_LBPHFaceRecognizer_create()")
        else:
            print("❌ No se encontró un reconocedor facial compatible")
            print("💡 Ejecuta: pip install opencv-contrib-python==4.8.1.78")
            return False
    except Exception as e:
        print(f"❌ Error al crear reconocedor: {e}")
        return False
    
    # Obtener imágenes y etiquetas
    faces, Ids = getImagesAndLabels("TrainingImage")
    
    if len(faces) == 0:
        print("❌ No se encontraron imágenes válidas para entrenar")
        return False
    
    print(f"📊 Entrenando con {len(faces)} imágenes y {len(set(Ids))} estudiantes...")
    
    # Entrenar el modelo
    try:
        recognizer.train(faces, np.array(Ids))
        
        # Guardar el modelo
        save_path = "TrainingImageLabel" + os.sep + "Trainner.yml"
        recognizer.save(save_path)
        
        print("=" * 50)
        print(f"✅ Modelo entrenado y guardado exitosamente!")
        print(f"📁 Ubicación: {save_path}")
        print(f"👥 Estudiantes entrenados: {len(set(Ids))}")
        print("=" * 50)
        return True
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return False

if __name__ == "__main__":
    TrainImages()