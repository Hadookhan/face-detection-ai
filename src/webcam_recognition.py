import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_recognizer.xml")
LABELS_PATH = os.path.join(BASE_DIR, "labels.npy")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def load_label_map(path):
    obj = np.load(path, allow_pickle=True).item()
    return {int(k): v for k, v in obj.items()}

def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("[ERROR] Model or labels not found. Run train_recognizer.py first.")
        return

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    label_map = load_label_map(LABELS_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            label_id, confidence = recognizer.predict(face_roi)
            name = label_map.get(label_id, "Unknown")

            text = f"{name} ({confidence:.0f})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow("OpenCV Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
