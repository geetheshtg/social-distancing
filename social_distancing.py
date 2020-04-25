from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import cv2 
import imutils 
from scipy.spatial import distance as dist

hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

while cap.isOpened():
    stack_x = []
    stack_y = []
    stack_x_print = []
    stack_y_print = []
    global D
    ret, image = cap.read()
    
    if ret:
        image = imutils.resize(image,width=min(600, image.shape[1]))

    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)
    
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    if len(pick) == 0:
        pass
    else:
        for i in range(0,len(pick)):
            x1 = pick[i][0]
            y1 = pick[i][1]
            x2 = pick[i][0] + pick[i][2]
            y2 = pick[i][1] + pick[i][3]

            mid_x = int((x1+x2)/2)
            mid_y = int((y1+y2)/2)
            stack_x.append(mid_x)
            stack_y.append(mid_y)
            stack_x_print.append(mid_x)
            stack_y_print.append(mid_y)

            image = cv2.circle(image, (mid_x, mid_y), 3 , [255,0,0] , -1)
            image = cv2.rectangle(image , (x1, y1) , (x2,y2) , [0,255,0] , 2)

        if len(pick) == 2:
            D = int(dist.euclidean((stack_x.pop(), stack_y.pop()), (stack_x.pop(), stack_y.pop())))
            image = cv2.line(image, (stack_x_print.pop(), stack_y_print.pop()), (stack_x_print.pop(), stack_y_print.pop()), [0,0,255], 2)
        else:
            D = 0

        if D<250 and D!=0:
            image = cv2.putText(image, "!!MOVE AWAY!!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,2, [0,0,255] , 4)

        image = cv2.putText(image, str(D/10) + " cm", (300, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255, 0, 0) , 2, cv2.LINE_AA)

        cv2.imshow('Frame' , image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
cv2.destroyAllWindows()