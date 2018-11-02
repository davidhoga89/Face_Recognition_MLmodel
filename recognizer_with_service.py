import cv2
import numpy as np
import os
import logging as log
import datetime as dt
import requests
from time import sleep

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

log.basicConfig(filename='webcam.log',level=log.INFO) # Creates log file
face_counter = 0 # variable for logging file

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
user_id = 0
#url for the WebApp
app_url = 'http://35.231.105.132/notification/user/'

# names related to ids: e.g. David Hoyos is id 1 when data_gathering and displayed with diff ID
names = ['none', 'Davidhoga', 'Alejandro', 'Adriana', 'Paola', 'DOwen','Jorge','Adolfo','Next']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size with a factor (0.15) to be recognized as a face
# higher factor means person needs to get closer to camera for detection
minW = 0.15*cam.get(3) #gets the width of the window
minH = 0.15*cam.get(4) #gets the height of the window

while True:
    if not cam.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray, # Greyscale image input
        scaleFactor = 1.2, # Param specifying how much the img size is reduced at each image scale
        minNeighbors = 5, # Param, with a higher value gives lower false positives
        minSize = (int(minW), int(minH)), # min rectangle to be considered a face
       )
    #this loop needs modification in order to just recognize face once and send id to the cloud service
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (200, 200, 75), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less than 100 | "0" is perfect match
        if (confidence < 100):
            user_name = names[id] # assigns matched userID with name from the array names (corresponding to its index in it)
            confidence = "  {0}%".format(round(100 - confidence))
            #prints in console and posts service everytime there is a new face recognized
            if (user_id != id):
                user_id = id
                print('Recognized User #' + str(user_id) + ': ' + user_name)
                #req = requests.post(app_url + str(user_id)) #deploys service to show events from user_id
        else:
            user_name = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            print('Unknown Visitor')

        cv2.putText(img, str(user_name), (x+5,y-5), font, 1, (200,200,200), 2)
        cv2.putText(img, str(confidence), (x+50,y+h-5), font, 0.75, (255,255,0), 1)

    # Fills Log file with faces detected and time_stamp
    if face_counter != len(faces):
        face_counter = len(faces)
        log.info("face detected-> "+str(len(faces))+" | "+ "ID#_"+str(user_id)+": "+user_name+" at "+str(dt.datetime.now()))

    cv2.imshow('camera',img)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print(app_url + str(user_id))
#req = requests.post(app_url + str(user_id))
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
