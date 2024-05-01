import cv2
import numpy as np
import os 
import winsound

def draw_found_faces(detected, image, color: tuple):
    for (x, y, width, height) in detected:
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            color,
            thickness=2
        )
        
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0

# names related to ids: example ==> Assef: id=1,  etc
names = ['Unknown', 'Assef', 'Abdullah', 'Ali', 'Mohammad', 'Mahdi'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    detected_eyes = eye_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=4)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
        draw_found_faces(detected_eyes, img, (0, 255, 0))

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (0,255,0), 3)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (0,255,0), 1) 
            
        else:
            id = names[0]
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 3)
            draw_found_faces(detected_eyes, img, (0, 0, 255))

            winsound.PlaySound('Alarm/alert.wav',winsound.SND_ASYNC)
            confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (0,0,255), 3)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (0,0,255), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
