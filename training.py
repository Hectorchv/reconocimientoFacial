import cv2
import numpy as np
import os
from sklearn.utils import parallel_backend
import joblib

datasetPath = "dataset"
people = os.listdir(datasetPath)

print("Lista de persona", people)


labels = []
faces = []
label = 0

for name in people:
    datasetPerson = datasetPath + '/' + name
    print("Leyendo dataset de ", name)
    for photo in os.listdir(datasetPerson):
        labels.append(label)

        faces.append(cv2.imread(datasetPerson + '/' + photo, 0))
        imagen = cv2.imread(datasetPerson + '/' + photo, 0)
        cv2.imshow('imagen', imagen)
        cv2.waitKey(10)
    label += 1

n_jobs = -1

cv2.destroyAllWindows()
print("Numero de personas: ", label)
print("Numero de imagenes: ", np.count_nonzero(np.array(labels)))

#Entrenamiento del modelo

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
print("Entrenando modelo")
with parallel_backend('threading', n_jobs=n_jobs):
    faceRecognizer.train(faces, np.array(labels))

faceRecognizer.write('ModelFacesFrontalCustom.xml')
print("Modelo generado")