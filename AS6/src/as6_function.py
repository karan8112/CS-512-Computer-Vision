# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 02:24:34 2018

@author: karan
"""

import numpy as np
from numpy import mean,std




#normlaise the list of points we enter and return the M matrix
def normalise(pts1):
    a,b,z =std(pts1, axis=0)
    sig=np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            if i==j==0:
                sig[i,j] = 1/a
            elif i==j==1:
                sig[i,j] = 1/b
            elif i==j==2:
                sig[i,j]=1
    c,d,q=mean(pts1, axis=0)
    mu = np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            if i==j==0:
                mu[i,j] = 1      
            elif i==j==1:
                mu[i,j] = 1
            elif i==j==2:
                mu[i,j]=1 
    mu[0,2]=-c
    mu[1,2]=-d
    M =np.matmul(sig,mu)
    return M 


# make a f matrix by using normalized points of left and right image points make sure that the rank is 2. 
def F_matrix(norm_left,norm_right):
    total_matrix = []
    for i,j in zip(norm_left,norm_right):
        lx,ly,d = i
        rx,ry,e= j
        a= lx*rx,ly*rx,rx,lx*ry,ly*ry,ry,lx,ly,1
        matrix= []
        matrix.extend(a)
        total_matrix.append(matrix)
    U, s, V = np.linalg.svd(total_matrix)
    m = V.T[:,-1].reshape(3,3)
    U, s, V = np.linalg.svd(m)
    S = np.diag(s)
    row=S.shape[0]-1
    col=S.shape[1]-1
    S[row,col]=0
    return (np.dot(U,np.dot(S,V)))


#find epipolar points
def epipolar(F):
    U, s, V = np.linalg.svd(F)
    left_epipolar_point = V.T[:,-1]
    right_epipolar_point = U[:,-1]
    return left_epipolar_point,right_epipolar_point