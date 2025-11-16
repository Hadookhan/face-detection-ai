import face_recognition
from pathlib import Path

here = Path(__file__).resolve().parent
image1_path = here.parent / "test-images" / "face1.jpg"
image3_path = here.parent / "test-images" / "face3.jpg"

# Load images
img1 = face_recognition.load_image_file(image1_path)
img3 = face_recognition.load_image_file(image3_path)

# Get encodings of each, and take first encoded face from each
enc1 = face_recognition.face_encodings(img1)[0]
enc3 = face_recognition.face_encodings(img3)[0]

# Compare encodings
results = face_recognition.compare_faces([enc1], enc3)
distance = face_recognition.face_distance([enc1], enc3)[0]

print("Same person? ", results[0])
print("Distance: ", distance)