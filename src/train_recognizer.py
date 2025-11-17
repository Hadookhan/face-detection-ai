import cv2
import os
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "face_recognizer.xml")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "labels.npy")

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def load_images_and_labels(dataset_dir):
    face_images = []
    labels = []
    label_map = {}
    current_label_id = 0

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"[INFO] Processing: {person_name}")
        label_map[current_label_id] = person_name

        for filename in os.listdir(person_dir):
            filepath = os.path.join(person_dir, filename)
            img = cv2.imread(filepath)
            if img is None:
                print(f"[WARN] Could not read {filepath}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            if len(faces) == 0:
                print(f"[WARN] No face found in {filepath}")
                continue

            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                face_images.append(face_roi)
                labels.append(current_label_id)

        current_label_id += 1

    return face_images, np.array(labels), label_map

def main():
    print("[INFO] Loading dataset from", DATASET_DIR)
    faces, labels, label_map = load_images_and_labels(DATASET_DIR)

    if len(faces) == 0:
        print("[ERROR] No faces found in dataset. Check your dataset/ structure.")
        return

    print(f"[INFO] Training on {len(faces)} face images...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)

    recognizer.write(MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    np.save(LABELS_PATH, label_map)
    print(f"[INFO] Label map saved to {LABELS_PATH}")

if __name__ == "__main__":
    main()
