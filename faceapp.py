import cv2
# import matplotlib.pyplot as plt

# capture=cv2.VideoCapture(0) #To open camera 1
# capture=cv2.imread('girl.png')
# if (capture.isOpened()==False):
# 	print("Sorry")
# while(capture.isOpened()):

    # Load trained cascade classifier
# ret,frame=capture.read()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
 
 
    # Read the given image
color_image = cv2.imread('image.jpg')
 
    # Convert color image into grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
 
    # Detect faces ROI
    #Syntax: Classifier.detectMultiScale(input image, Scale Factor , Min Neighbors)
faces = face_cascade.detectMultiScale(gray_image, 1.1, 5) 
    # print(faces)
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
cv2.waitKey(3000)
# color_image.release()
cv2.destroyAllWindows()
# pics=['multi2.png','girl.png']
# import random
# detect_pic(random.choice(pics))
# import pickle
# pickle.dump(detect_pic('multi2.png'),open('model.pkl','w'))
