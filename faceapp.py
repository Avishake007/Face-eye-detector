import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
 
 
    # Read the given image
color_image = cv2.imread('image.jpg')
 
    # Convert color image into grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    #Syntax: Classifier.detectMultiScale(input image, Scale Factor , Min Neighbors)
faces = face_cascade.detectMultiScale(gray_image, 1.1, 5) 
    # print(faces)
eyes=eye_cascade.detectMultiScale(gray_image, 1.1, 5) 
    # Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 4)
    # Draw rectangle around the eyes
for (x, y, w, h) in eyes:
    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
    # Show image
cv2.imshow('Image', color_image)
cv2.imwrite('detect.png',color_image)
cv2.waitKey(3000)
cv2.destroyAllWindows()
