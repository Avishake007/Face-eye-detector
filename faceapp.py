import os.path
from os import path
import cv2
def face_eye_detection():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
    pat=input("Enter the path of the image : ")
    #Checks whether the entered path is valid
    if (path.exists(pat)) : 
        # Read the given image
        color_image = cv2.imread('image.jpg')
        # Convert color image into grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # Detect faces ROI
        #Syntax: Classifier.detectMultiScale(input image, Scale Factor , Min Neighbors)
        faces = face_cascade.detectMultiScale(gray_image, 1.1, 5) 
        a=str(len(faces))#Storing the number of detected faces
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(color_image,a+"faces detected",(15,350), font, 2,(0,0,0),2,cv2.LINE_AA)
        eyes=eye_cascade.detectMultiScale(gray_image, 1.1, 5) 
            # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 4)
                # print(x,y,w,h)
        for (x, y, w, h) in eyes:
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Show image
        cv2.imshow('Image', color_image)
        cv2.imwrite('detect.png',color_image)
        # viewImage(color_image,"hi")
        cv2.waitKey(30000)
        # color_image.release()
        cv2.destroyAllWindows()
    else:
        print("The entered path is invalid")
if __name__=="__main__":
    face_eye_detection()
