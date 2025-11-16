import face_recognition
import cv2
from pathlib import Path

here = Path(__file__).resolve().parent
image1_path = here.parent / "test-images" / "face1.jpg"
image2_path = here.parent / "test-images" / "face2.jpg"
group_path = here.parent / "test-images" / "group.jpg"

# Load all known faces
known_images = [
    ("Hadi", image1_path),
    ("Shiza", image2_path)
]

known_face_encodings = []
known_face_names = []

for name, filename in known_images:
    img = face_recognition.load_image_file(filename)
    encoding = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

# 2. Loading an unknown image (group photo)
unknown_image = face_recognition.load_image_file(group_path)
unknown_face_locations = face_recognition.face_locations(unknown_image)
unknown_face_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)

# Convert unkown image from RGB to BGR to be drawn
image_bgr = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

# Compare each unknown face to known faces to find a match
for (top, right, bottom, left), face_encoding in zip(unknown_face_locations, unknown_face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # Use distance to find best match
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = face_distances.argmin()
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Draw boxes and label
    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0,255,0), 2)
    cv2.rectangle(image_bgr, (left, top), (right, bottom+20), (0,0,255), cv2.FILLED)
    cv2.putText(image_bgr, name, (left+2,bottom+15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1)

cv2.imshow("Recognised Faces ", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
