# Count-number-of-Faces-using-Python-of-image

In this article, we will use image processing to detect and count the number of faces. We are not supposed to get all the features of the face. Instead, the objective is to obtain the bounding box through some methods i.e. coordinates of the face in the image, depending on different areas covered by the number of the coordinates, number faces that will be computed.

Required libraries:
OpenCV library in python is a computer vision library, mostly used for image processing, video processing, and analysis, facial recognition and detection, etc.
Dlib library in python contains the pre-trained facial landmark detector, that is used to detect the (x, y) coordinates that map to facial structures on the face.
Numpy is a general-purpose array-processing package. It provides a high-performance multidimensional array object and tools for working with these arrays.

import cv2
from google.colab.patches import cv2_imshow
# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load an image or start a video capture
# For an image, 
img = cv2.imread('/content/file.enc')

# Detect faces in the image
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Get the number of faces found
num_faces = len(faces)

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with faces
cv2_imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the number of faces found
print("Number of faces detected: " + str(num_faces))
![Uploading image.png…]()
Number of faces detected: 1
