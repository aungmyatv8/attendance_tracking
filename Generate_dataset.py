import cv2
import os
from pyzbar import pyzbar


# Function to capture face images using a webcam
def capture_face_images(user_name, num_images=30):
    user_dir = f'face_dataset/{user_name}'
    os.makedirs (user_dir, exist_ok=True)
    face_classifier = cv2.CascadeClassifier ("haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture (0)

    face_count = 0
    while face_count < num_images:
        ret, frame = cap.read ( )
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale (gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cropped_face = frame[y:y + h, x:x + w]
            face_count += 1
            face = cv2.resize (cropped_face, (200, 200))
            face = cv2.cvtColor (face, cv2.COLOR_BGR2GRAY)
            face_path = os.path.join (user_dir, f'face_{face_count}.jpg')
            cv2.imwrite (face_path, face)
            cv2.putText (face, str (face_count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            print (f"Captured face image {face_count + 1} for {user_name}")
            cv2.imshow ("Cropped face", face)

        if cv2.waitKey (1) == 13 or int (face_count) == 30:  # 13 is the ASCII character of Enter
            break

    cap.release ( )
    cv2.destroyAllWindows ( )
    print (f"Face image capture complete for {user_name}.")


# Function to extract QR code from an existing image
def extract_qr_code_from_image(user_name, image_path):
    qr_code_dir = 'qr_dataset'
    user_qr_dir = os.path.join (qr_code_dir, user_name)
    os.makedirs (user_qr_dir, exist_ok=True)

    frame = cv2.imread(image_path)
    if frame is None:
        print (f"Failed to load image: {image_path}")
        return

    decoded_objects = pyzbar.decode(frame)
    if not decoded_objects:
        print (f"No QR code detected in the image: {image_path}")
        return

    for obj in decoded_objects:
        if obj.type == "QRCODE":
            (x, y, w, h) = obj.rect
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                print ("Invalid QR code coordinates, skipping...")
                continue

            qr_code_image = frame[y:y + h, x:x + w]
            qr_code_path = os.path.join (user_qr_dir, 'qr.png')
            cv2.imwrite(qr_code_path, qr_code_image)
            print (f"Extracted and saved QR code image for {user_name}: {qr_code_path}")
            return

    print (f"No valid QR code found in the image: {image_path}")


# Main script
user_name = input ("Enter user name: ")
capture_face_images (user_name)
image_path = input ("Enter path to the image containing the QR code: ")
extract_qr_code_from_image (user_name, image_path)
