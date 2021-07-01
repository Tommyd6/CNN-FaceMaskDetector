import cv2
import os
import random
file = 'train/with_mask'  # directory containing images
while os.listdir(file):  # loop till directory empty
    random_file = random.choice(os.listdir(file))  # file from directory
    image = cv2.imread(file+'/'+random_file)  # changes file to image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # gets a grey image copy

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # harrcascade used
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )  # detects faces in the image
    # creates image of just face
    for (x, y, w, h) in faces:
        roi_color = image[y:y + h, x:x + w] # gets new image
        cv2.imwrite('faces_'+random_file, roi_color)

    # removes file from directory
    os.remove(file+'/'+random_file)
