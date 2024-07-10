import cv2
import os
import numpy as np
from PIL import Image


def train_classifier(dataset_dir):
    recognizer = cv2.face.LBPHFaceRecognizer_create( )
    faces = []
    ids = []
    user_id = 0

    # Iterate through each user directory in the dataset directory
    for user_name in os.listdir(dataset_dir):
        print("user name", user_name)
        user_dir = os.path.join(dataset_dir, user_name)
        if not os.path.isdir (user_dir):
            continue

        user_id += 1  # Increment user ID for each user
        for image_name in os.listdir(user_dir):
            image_path = os.path.join(user_dir, image_name)
            print("image path", image_path)
            gray_img = Image.open(image_path).convert ('L')
            img_np = np.array (gray_img, 'uint8')
            faces.append (img_np)
            ids.append (user_id)

    # Convert ids to numpy array
    ids = np.array(ids)
    print("ids", ids)
    # Train the recognizer
    recognizer.train(faces, ids)

    # Save the trained model
    recognizer.save('trainer.yml')
    print ("Training complete.")


dataset_directory = "./face_dataset"
train_classifier(dataset_directory)
