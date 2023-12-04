import cv2
import numpy
import os

face_cascade = cv2.CascadeClassife("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(1)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rostro = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in rostros:
        cv2.rectangle(img, (x,y), (x+w, y+h,) (0,255,0), 3)
    cv2.imgshow('img', img)
    if cv2.waitKey(30):
        break
cap.release()