
import cv2
import numpy as np
import os

# Load the Haar cascade for face detection
haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Path to the dataset directory
datasets = 'dataset'

print('Training...')

# Initialize lists for storing images, labels, names, and IDs
(images, labels, names, id) = ([], [], {}, 0)

# Iterate through the dataset directory
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        # Map each name to an ID
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        # Iterate through images in each subdirectory
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            # Read images and assign labels
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

# Convert lists to NumPy arrays
(images, labels) = [np.array(lis) for lis in [images, labels]]

# Define the desired width and height for face resizing
(width, height) = (130, 100)

# Initialize the LBPH Face Recognizer
model = cv2.face.LBPHFaceRecognizer_create()

# Train the model with the images and labels
model.train(images, labels)

# Initialize the webcam
webcam = cv2.VideoCapture(0)

cnt = 0
threshold = 100

# Main loop for real-time face recognition
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # Predict the ID of the person in the face region
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Display the recognized name if confidence is below the dynamic threshold
        if prediction[1] <= threshold:
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, 'Unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("unKnown.jpg", im)
                cnt = 0
    cv2.imshow('FaceRecognition', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()


