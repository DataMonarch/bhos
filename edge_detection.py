import cv2
import numpy as np

#Capture livestream video content from camera 0
cap = cv2.VideoCapture(0)

kernel_size = 5

while(1):

    # Take each frame
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(kernel_size, kernel_size),0)

    # Convert to HSV for simpler calculations
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calcution of Sobelx
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)

    # Calculation of Sobely
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)

    # Calculation of Laplacian
    laplacian = cv2.Laplacian(frame,cv2.CV_64F)

    cv2.imshow('sobelx',sobelx)
    cv2.imshow('sobely',sobely)
    cv2.imshow('laplacian',laplacian)
    if cv2.waitKey(1) & 0xFF == 'q':
        break

cv2.destroyAllWindows()

#release the frame
cap.release()
