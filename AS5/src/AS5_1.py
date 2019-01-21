# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


img = cv2.imread("C:\Users\karan\.spyder-py3\ChessBoard.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (7,7), None)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)
    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (7,7), corners2, ret)
    print("1")
    cv2.imshow('img',img)
    cv2.waitKey()

cv2.destroyAllWindows()
print(imgpoints)

#write points in the file
with open('C:/Users/karan/.spyder-py3/testing_file.txt', 'w') as f:
    for i,j in zip(objp, corners2.tolist()):
        #f.write("%s \n" % point)
        f.write(str(i[0]) + "," + str(i[1]) + "," + str(i[2]) + "," + str(j[0][0]) + "," + str(j[0][1]))
        f.write("\n")
    f.close()
        
