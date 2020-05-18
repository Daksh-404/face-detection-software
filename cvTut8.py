import cv2 as cv
import numpy as np

faceCascade=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
img=cv.imread("girls.jpg")
img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(img_gray,1.1,4)

for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

#the face recognition works on face cascad xml files created
# and compiled over the time by openCV
#this is viola and Jones method, for face recognition

cv.imshow("Output",img)
cv.waitKey(0)
