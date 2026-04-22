import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread

def getImagesAndLabels(path):
    # path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    # empty ID list
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        # Asegúrate de que el formato del nombre de la imagen sea 'nombre.ID.extension'
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath) # Opcional: no lo estás usando aquí, pero no hace daño
    
    faces, Id = getImagesAndLabels("TrainingImage")
    
    # 1. Iniciar el contador visual en un hilo separado (sintaxis correcta usando 'args')
    Thread(target=counter_img, args=("TrainingImage",)).start()
    
    # 2. Entrenar el modelo en el hilo principal (esto tomará unos segundos)
    recognizer.train(faces, np.array(Id))
    
    # 3. Guardar el modelo SOLO DESPUÉS de que la línea anterior termine
    save_path = "TrainingImageLabel" + os.sep + "Trainner.yml"
    recognizer.save(save_path)
    
    print("\nModelo entrenado y guardado exitosamente.")


def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for _ in imagePaths:
        print(str(imgcounter) + " Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1