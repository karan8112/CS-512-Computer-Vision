# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:33:13 2018

@author: karan
"""

import cv2
import numpy as np


image_point = []
origin_point = []
num_lines = 0

def main():
    #read data from file
    filename = input("Input filepath:")
    with open(filename,"r") as r_f:
	for line in r_f.readlines():
		data = line.split(",")
		if len(data) < 5: 
			continue
		origin_point.append((float(data[0]), float(data[1]), float(data[2])))
		image_point.append((float(data[3]), float(data[4])))
    global num_lines
    num_lines = sum(1 for line in open(filename))
    M = projection_matrix(num_lines)
    calculate_parameters(M)
    calculate_mse(M)
    
def calculate_parameters(M):
    #intrinsic parameter
    print("Intrinsic Parameter\n")
    a1 = [ M[0][0], M[0][1], M[0][2] ]
    a2 = [ M[1][0], M[1][1], M[1][2] ]
    a3 = [ M[2][0], M[2][1], M[2][2] ]
    b = [  M[0][3], M[1][3], M[2][3] ]
    
    r_mag = 1/np.linalg.norm(a3)
    u0 = r_mag * r_mag * np.dot(a1,a3)
    v0 = r_mag * r_mag * np.dot(a2,a3)
    print("(u0,v0): ",u0,v0)
    print("\n")

    alphaV = np.sqrt(r_mag * r_mag * np.dot(a2,a2 ) - v0 * v0)
    s = np.dot( (r_mag * r_mag * r_mag * r_mag) / alphaV * np.cross(a1, a3), np.cross(a2, a3) )
    print("s: ", s)
    print("\n")
    alphaU = np.sqrt(r_mag * r_mag * np.dot(a1,a1) - s * s - u0 * u0)
    print("(alphaU, alphaV): ", alphaU, alphaV)
    print("\n")
    Kstar = [ [alphaU, s, u0], [0, alphaV, v0], [0, 0, 1] ]
    print("K*: ", Kstar)
    print("\n\n\n")
    
    
    #extrinsic parameter
    print("Extrinsic Parameter\n")
    e = 1
    if b[2] < 0:
        e = 1
    Tstar = e * r_mag * np.matmul(np.linalg.inv(Kstar) , b)
    print("T*: ", Tstar)
    print("\n")
    r3 = e * r_mag * np.transpose(a3)
    r1 = r_mag * r_mag / alphaV * np.cross(a2, a3)
    r2 = np.cross(r3, r1)
    Rstar = np.matrix([np.transpose(r1), np.transpose(r2), np.transpose(r3)])
    print("R*: ", Rstar)
    print("\n\n\n")
    

def calculate_mse(M):
    p_image_point = []
    for i in range(0, num_lines):
         p = np.matmul(M, [ origin_point[i][0], origin_point[i][1], origin_point[i][2], 1])
         p_image_point.append((p[0]/p[2], p[1]/p[2]))
    err = 0
    for i in range(0, num_lines):
	    err = err + (image_point[i][0] - p_image_point[i][0]) ** 2
	    err = err + (image_point[i][1] - p_image_point[i][1]) ** 2
    print("Mean Square Error: ", err)
    
    
def projection_matrix(num_lines):
    #estimate projection matirix M
    A = []
    print(num_lines)
    for i in range(0,num_lines):
        A.append([ origin_point[i][0], origin_point[i][1], origin_point[i][2], 1, 0, 0, 0, 0, -1 * image_point[i][0] * origin_point[i][0], -1 * image_point[i][0] * origin_point[i][1], -1 * image_point[i][0] * origin_point[i][2], -1 * image_point[i][0] * 1 ])
        A.append([ 0, 0, 0, 0, origin_point[i][0], origin_point[i][1], origin_point[i][2], 1, -1 * image_point[i][1] * origin_point[i][0], -1 * image_point[i][1] * origin_point[i][1], -1 * image_point[i][1] * origin_point[i][2], -1 * image_point[i][1] * 1 ])
    U, D, V = np.linalg.svd(A, full_matrices=True)
    print("ssds")
    print(V[11]) 
    M = np.split(V[11], 3)
    print("M: ", M)
    print()
    return M



if __name__ == '__main__':
    main()