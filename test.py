import cv2
import numpy as np
from keras.models import load_model

model = load_model("./model-018.model")  # loads in model
# text and colours for out of CNN
results = {2: 'without mask', 1: 'mask',0: 'Mask wrong'}
GR_dict = {2: (0, 0, 255), 1: (0, 255, 0), 0: (0, 255, 255)}
rect_size = 5
# sets uo webcam
cap = cv2.VideoCapture(0)
haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # haarcascade used
while True:
    (rval, im) = cap.read()
    im = cv2.flip(im, 1, 1)  # flips image so webcam looks like a mirror
    # resize image for object detection for speed
    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    # detects face
    faces = haarcascade.detectMultiScale(rerect_size)
    # this section send face image into CNN and makes the rectangles
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]

        face_img = im[y:y + h, x:x + w]
        # gets face image ready for CNN
        rerect_sized = cv2.resize(face_img, (150, 150))
        normalized = rerect_sized / 255.0
        reshaped = np.reshape(normalized, (1, 150, 150, 3))
        reshaped = np.vstack([reshaped])
        # Send face image into CNN
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]
        # makes rectangles
        cv2.rectangle(im, (x, y), (x + w, y + h), GR_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), GR_dict[label], -1)
        # makes text
        cv2.putText(im, results[label], (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imshow('LIVE', im)
    # turns off webcam when esc pressed
    key = cv2.waitKey(10)
    if key == 27:
        break
# release webcam and close program
cap.release()
cv2.destroyAllWindows()
