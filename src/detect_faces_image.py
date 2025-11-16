import face_recognition
import cv2
from pathlib import Path

here = Path(__file__).resolve().parent
image_path = here.parent / "test-images" / "face2.jpg"
detected_path = here.parent / "detected" / "face2_detected.jpg"
image = face_recognition.load_image_file(image_path) # loads image

face_locations = face_recognition.face_locations(image) # returns location of each face found in photo. location=(top, right, bottom, left)

print(f"Found {len(face_locations)} face(s).")

image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Converts RGB to BGR for OpenCV

# Draws boxes
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

# Shows image using cv2
cv2.imshow("Faces", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Saves image to new file
cv2.imwrite(detected_path, image_bgr)