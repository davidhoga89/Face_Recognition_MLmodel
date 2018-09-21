import numpy as np
import cv2
import logging as log
import datetime as dt
from time import sleep

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
log.basicConfig(filename='webcam.log',level=log.INFO) # Creates log file

video_capture = cv2.VideoCapture(0) # acquires webcam video
video_capture.set(3,640) # set Width
video_capture.set(4,480) # set Height
face_counter = 0 # variable for logging file

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    font = cv2.FONT_HERSHEY_SIMPLEX
    # this variable below will be the name/# of the new/returning user
    faceID = "ID: David Hoyos"

    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray, # Greyscale image input
        scaleFactor=1.2, # Param specifying how much the img size is reduced at each image scale
        minNeighbors=10, # Param, with a higher value gives lower false positives
        minSize=(75, 75) # Minimum rectangle size to be considered a face, far detection requieres lower #s
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(200, 200, 75),2) # Face frame with position, color & line width
        cv2.putText(img,faceID, # Text of the image "faceID"
        (x,y),font, # Position & font
        0.75, # Size of text
        (200,200,200),2) # Text color & line width

        roi_gray = gray[y:y+h, x:x+w] # displays position
        roi_color = img[y:y+h, x:x+w] # displays RGB colors

    # Fills Log file with faces detected and time_stamp
    if face_counter != len(faces):
        face_counter = len(faces)
        log.info("face detected: "+str(len(faces))+" | "+faceID+" at "+str(dt.datetime.now()))

    # Shows the resulting video capture with face detection
    cv2.imshow('Face Detector',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

video_capture.release()
cv2.destroyAllWindows()
