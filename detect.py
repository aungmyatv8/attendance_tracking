import cv2
import numpy as np
from PIL import Image
import os
from pyzbar import pyzbar
import RPi.GPIO as GPIO
import time

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)  # Buzzer
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Tact Switch

# Load the trained face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Dictionary to hold user data
user_data = {}

# Load user data
def load_user_data(face_dataset_dir, qr_dataset_dir):
    user_id = 0
    for user_name in os.listdir(face_dataset_dir):
        user_face_dir = os.path.join(face_dataset_dir, user_name)
        user_qr_dir = os.path.join(qr_dataset_dir, user_name)

        if not os.path.isdir(user_face_dir) or not os.path.isdir(user_qr_dir):
            continue

        user_id += 1
        qr_code_path = os.path.join(user_qr_dir, 'qr.png')
        qr_code_image = cv2.imread(qr_code_path)
        qr_code_data = None

        if qr_code_image is not None:
            qr_codes = pyzbar.decode(qr_code_image)
            if len(qr_codes) > 0:
                qr_code_data = qr_codes[0].data.decode('utf-8')

        user_data[user_id] = {
            'name': user_name,
            'qr_code': qr_code_data,
            'authorized': qr_code_data is not None  # Assume authorization if QR code is present
        }

# Function to draw boundary around detected face
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        id, confidence = clf.predict(gray_img[y:y + h, x:x + w])

        if confidence < 100:
            confidence = int(100 * (1 - confidence / 300))
            if confidence > 70:
                name = user_data.get(id, {}).get('name', 'Unknown')
                if name != 'Unknown':
                    GPIO.output(18, GPIO.HIGH)  # Turn on buzzer for known user
                cv2.putText(img, name, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "Unknown", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Unknown", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

        coords = [x, y, w, h]
    return coords

# Function to recognize faces and QR codes
def recognize_faces_and_qr_codes():
    cap = cv2.VideoCapture(0)

    while True:
        if GPIO.input(23) == GPIO.LOW:  # Check if tact switch is pressed
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                user_id = None
                user_name = "Unknown"
                authorized = False

                # Detect QR codes
                qr_codes = pyzbar.decode(frame)
                for qr_code in qr_codes:
                    (x, y, w, h) = qr_code.rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    qr_code_text = qr_code.data.decode('utf-8')

                    # Check if the QR code matches any user's QR code
                    for id, data in user_data.items():
                        if data['qr_code'] == qr_code_text:
                            user_id = id
                            user_name = data['name']
                            authorized = data['authorized']
                            break

                    if user_id is not None:
                        status = "Authorized User" if authorized else "Unauthorized User"
                        cv2.putText(frame, f"{status}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        break

                # If a known user is identified by QR code, proceed to face recognition
                if user_id is not None:
                    coords = draw_boundary(frame, face_cascade, 1.1, 10, (255, 0, 0), "Face", recognizer)
                    print("coords", coords)

                # Show the frame with detected faces and QR codes
                cv2.imshow('Face and QR Code Recognition', frame)

                if cv2.waitKey(1) == 13:  # Press Enter to exit the recognition process
                    GPIO.output(18, GPIO.LOW)  # Turn off buzzer after processing
                    break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load user data from both datasets
load_user_data('face_dataset', 'qr_dataset')
recognize_faces_and_qr_codes()

# Cleanup GPIO
GPIO.cleanup()
