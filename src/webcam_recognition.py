import cv2
import face_recognition
import numpy as np
from pathlib import Path

here = Path(__file__).resolve().parent
me_path = here.parent / "test-images" / "face1.jpg"
friend_path = here.parent / "test-images" / "face2.jpg"

known_face_encodings = []
known_face_names = []

def add_known_face(name, filename):
    img = face_recognition.load_image_file(filename)
    enc = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(enc)
    known_face_names.append(name)

add_known_face("Me", me_path)
add_known_face("Friend", friend_path)

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resizing the frame to increase speed
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1] # Converts BGR to RGB

    # Detect faces + encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unkown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

    # Drawing the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom), (right, bottom + 20), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("Webcam Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
