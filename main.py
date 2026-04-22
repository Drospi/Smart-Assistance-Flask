import os  # os functions
import check_camera
import capture_image
import train_image
import recognize
import shutil


def title_bar():
    os.system('cls')
    print("\t***** Face Recognition Attendance System *****")


def mainMenu():
    title_bar()
    print()
    print(10 * "*", "WELCOME MENU", 10 * "*")
    print("[1] Check Camera")
    print("[2] Capture Faces")
    print("[3] Train Images")
    print("[4] Recognize & Attendance")
    print("[5] Auto Mail")
    print("[6] Quit")
    print("[7] Reset System") 
    while True:
        try:
            choice = int(input("Enter Choice: "))
            if choice == 1:
                checkCamera()
                break
            elif choice == 2:
                CaptureFaces()
                break
            elif choice == 3:
                Trainimages()
                break
            elif choice == 4:
                recognizeFaces()
                break
            elif choice == 5:
                os.system("py automail.py")
                break
                mainMenu()
            elif choice == 6:
                print("Thank You")
                break
            elif choice == 7:
                confirm = input("⚠️ Esto borrará TODOS los datos. ¿Continuar? (s/n): ")

                if confirm.lower() == 's':
                    resetSystem()
                else:
                    print("❌ Cancelado")
            else:
                print("Invalid Choice. Enter 1-4")
                mainMenu()
        except ValueError:
            print("Invalid Choice. Enter 1-4\n Try Again")
    exit

# camera test function from check camera.py file
def checkCamera():
    check_camera.camer()
    key = input("Enter any key to return main menu")
    mainMenu()

# image function form capture image.py file
def CaptureFaces():
    capture_image.takeImages()
    key = input("Enter any key to return main menu")
    mainMenu()

# train images from train_images.py file
def Trainimages():
    train_image.TrainImages()
    key = input("Enter any key to return main menu")
    mainMenu()

# recognize_attendance from recognize.py file
def recognizeFaces():
    recognize.recognize_attendence()
    key = input("Enter any key to return main menu")
    mainMenu()

def resetSystem():
    folders = [
        "TrainingImage",
        "TrainingImageLabel",
        "Attendance"
    ]

    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"[OK] Carpeta eliminada: {folder}")
        os.makedirs(folder)
        print(f"[OK] Carpeta creada: {folder}")

    print("\n🔥 Sistema reiniciado correctamente\n")

# main driver
mainMenu()

