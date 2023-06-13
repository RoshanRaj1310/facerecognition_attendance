import cv2
import numpy as np
import face_recognition

imgmessi= face_recognition.load_image_file('data/messi1.jpg')
imgmessi= cv2.cvtColor(imgmessi,cv2.COLOR_BGR2RGB)
imgtest= face_recognition.load_image_file('data/messi2.jpg')
imgtest= cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceloc=face_recognition.face_locations(imgmessi)[0]
encodemessi=face_recognition.face_encodings(imgmessi)[0]
cv2.rectangle(imgmessi,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloc=face_recognition.face_locations(imgtest)[0]
encodetest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodemessi],encodetest)
facedis=face_recognition.face_distance([encodemessi],encodetest)
print(results,facedis)
cv2.putText(imgtest,f'{results}{round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('img messi',imgmessi)
cv2.imshow('test messi',imgtest)
cv2.waitKey(0)