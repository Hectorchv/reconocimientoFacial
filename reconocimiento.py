import cv2
import numpy
import os

#Cargamos nuestro modelo
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("ModelFacesFrontalCustom.xml")


datasetPath = "dataset"
imgPath = os.listdir(datasetPath)
print(imgPath)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")




cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    copy = gray.copy()
    rostro = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in rostro:
        person = copy[y:y+h, x:x+h]
        person = cv2.resize(person, (150,150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(person)

        cv2.putText(img, '{}'.format(result), (x,y-5), 1, 1.3, (0,255,0),1, cv2.LINE_AA)
        if result[1] < 70:
            cv2.putText(img, '{}'.format(imgPath[result[0]]), (x,y-20),2,1,(255,0,0),1)
            cv2.rectangle(img, (x,y), (x+w, y+h,), (0,255,0), 3)
        else:
            cv2.putText(img, "Desconocido", (x,y-25),2,1,(255,0,0))
            cv2.rectangle(img, (x,y), (x+w, y+h,), (0,0,255), 3)
    
    cv2.imshow('img', img)
    if cv2.waitKey(30) == 27:
        break
cap.release()