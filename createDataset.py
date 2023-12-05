#!/usr/bin/python3

#Programa que crea el dataset de la persona a reconocer

import cv2
import numpy
import os
import time as tm


directorioDatos = "dataset/"
persona = "hector" #Nombre se la persona a detectar
dir = directorioDatos + persona

if os.path.exists(dir) and os.path.isdir(dir):
    print("Directorio existente")
else:
    print("Creando directorio: " + dir)
    os.mkdir(dir)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

n = 0

while True:
    _, img = cap.read()
    copy = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in face:
        file = copy[y+3:y+h-3,x+3:x+w-3]
        file = cv2.resize(file, (720,720), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(img, (x,y), (x+w, y+h,), (0,255,0), 3)
        filename = os.path.sep.join([dir, "{}.png".format(str(n).zfill(5))])
        cv2.imwrite(filename, file)
        n = n + 1


    cv2.imshow('img', img)

    if cv2.waitKey(30) == 27 or n >= 150:
        break
cap.release()
cv2.destroyAllWindows()