# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:13:46 2018

@author: karan
"""

import numpy as np
import cv2
color = np.random.randint(0,255,(10000,3))

def lucasKanade(new_frame,old_frame,feature_points,side,ksize):
    #convert color image to grayscale before applying filter
    old = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
    new = cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY)
    width,height = old.shape
    points = []
    system_matrix = []
    #apply filter
    a,b = np.gradient(old)
    c,d = np.gradient(new)
    old = cv2.GaussianBlur(old,(5,5),0)
    new = cv2.GaussianBlur(new,(5,5),0)
    Ix_ = (c-a)/2
    Iy_ = (d-b)/2
    It_ = new - old
    for x,y in feature_points:
        x,y = int(x),int(y)
        if x+side>width or x-side<0 or y+side>height or y-side<0:
            points.append((0,0))
            system_matrix.append(np.zeros((2,2)))
            continue
        Ix = Ix_[(x-side-1):(x+side),(y-side-1):(y+side)]
        Iy = Iy_[(x-side-1):(x+side),(y-side-1):(y+side)]
        It = It_[(x-side-1):(x+side),(y-side-1):(y+side)]
        a11 = np.sum(np.power(Ix.ravel(),2))
        a22 = np.sum(np.power(Iy.ravel(),2))
        a_off = np.sum(Ix.ravel()*Iy.ravel())
        A = np.array([[a11,a_off],[a_off,a22]])
        b11 = -(np.sum(Ix.ravel()*It.ravel()))
        b21 = -(np.sum(Iy.ravel()*It.ravel()))
        b = np.array([[b11],[b21]])
        Ainv = np.linalg.pinv(A)
        x = np.matmul(Ainv,b)
        points.append((x[0,0],x[1,0]))
        system_matrix.append(A)
    return points,system_matrix



def calculate_reliability(matrix):
    rel = []
    for i,x in enumerate(matrix):
        if np.all(x==0):
            rel.append(0)
        else:
            u,d,v = np.linalg.svd(x)
            r = d[1]/d[0]
            rel.append(r)
    return rel



def draw_vectors(frame,old_points,new_points,reliability):
    for i,(old,new) in enumerate(zip(old_points,new_points)):
        if old ==0 or new == 0:
            continue
        pts = color[i]
        pointt_hsv = cv2.cvtColor(np.uint8([[[pts[0],pts[1],pts[2]]]]),cv2.COLOR_BGR2HSV)
        pointt_hsv[0][0][2] = pointt_hsv[0][0][2]*reliability[i]
        c = cv2.cvtColor(pointt_hsv,cv2.COLOR_HSV2BGR)
        cv2.line(frame,(int(new[0]),int(new[1])),(int(old[0]),int(old[1])),c.tolist()[0][0],5)
    return frame