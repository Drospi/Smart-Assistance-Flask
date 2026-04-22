import csv
import cv2
import os

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def takeImages(Id=None, name=None):
    # Si no se pasan parámetros (modo consola), pedirlos por input
    if Id is None:
        Id = input("Enter Your Id: ")
    if name is None:
        name = input("Enter Your Name: ")
    
    Id = str(Id)
    
    if not is_number(Id):
        print("❌ Error: El ID debe ser numérico")
        return False
    
    if not name.isalpha():
        print("❌ Error: El nombre debe contener solo letras")
        return False
    
    # Crear carpetas si no existen
    if not os.path.exists("TrainingImage"):
        os.makedirs("TrainingImage")
    if not os.path.exists("StudentDetails"):
        os.makedirs("StudentDetails")
    
    # Inicializar cámara
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Error: No se pudo acceder a la cámara")
        return False
    
    harcascadePath = "haarcascade_default.xml"
    if not os.path.exists(harcascadePath):
        print("❌ Error: No se encuentra haarcascade_default.xml")
        return False
    
    detector = cv2.CascadeClassifier(harcascadePath)
    if detector.empty():
        print("❌ Error: No se pudo cargar el clasificador facial")
        return False
    
    sampleNum = 0
    print(f"📸 Iniciando captura para {name} (ID: {Id})")
    print("💡 Instrucciones:")
    print("   - Mantén tu rostro frente a la cámara")
    print("   - Se capturarán 100 imágenes automáticamente")
    print("   - Presiona 'q' para cancelar")
    print("=" * 50)
    
    while True:
        ret, img = cam.read()
        if not ret:
            print("Error al leer de la cámara")
            break
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (10, 159, 255), 2)
            sampleNum += 1
            
            # Guardar la imagen
            filename = f"TrainingImage{os.sep}{name}.{Id}.{sampleNum}.jpg"
            cv2.imwrite(filename, gray[y:y+h, x:x+w])
            
            # Mostrar progreso
            cv2.putText(img, f"Capturando: {sampleNum}/100", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 159, 255), 2)
        
        cv2.putText(img, f"Estudiante: {name} (ID: {Id})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Imagenes: {sampleNum}/100", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "Presiona 'q' para cancelar", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Capturando Rostros', img)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            print("\n⚠️ Captura cancelada por el usuario")
            break
        elif sampleNum >= 100:
            print("\n✅ Captura completada!")
            break
    
    cam.release()
    cv2.destroyAllWindows()
    
    if sampleNum > 0:
        # Guardar en StudentDetails.csv
        row = [Id, name]
        file_exists = os.path.isfile("StudentDetails"+os.sep+"StudentDetails.csv")
        
        with open("StudentDetails"+os.sep+"StudentDetails.csv", 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            if not file_exists or os.path.getsize("StudentDetails"+os.sep+"StudentDetails.csv") == 0:
                writer.writerow(['Id', 'Name'])
            writer.writerow(row)
        
        print("=" * 50)
        print(f"✅ Éxito! Se capturaron {sampleNum} imágenes")
        print(f"👤 Estudiante: {name} (ID: {Id})")
        print("💡 Siguiente paso: Ve a la sección 'Train' para entrenar el modelo")
        print("=" * 50)
        return True
    else:
        print("❌ No se capturaron imágenes")
        return False

if __name__ == "__main__":
    takeImages()