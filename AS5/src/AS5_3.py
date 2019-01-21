import numpy as np
import random


#default parameter if config file not found
d = 6
n = 10
w = 0.5
p = 0.99
t = 0.25

#global parameter
image_point = []
origin_point = []
num_lines = 0
   
#read data from given input file where first  3 columns contain world data and last 2 columns contain image data 
def read_data():
    global num_lines
    filename = input("Data file path : ")
    with open(filename,"r") as f:
        num_lines = sum(1 for line in open(filename))
        for line in f.readlines():
            line = line.strip('\n')
            data = line.split(",")
            if len(data) < 5:
                continue
            origin_point.append((float(data[0]), float(data[1]), float(data[2])))
            image_point.append((float(data[3]), float(data[4])))


#read RENSAC config file
def read_config_parameter():
    global d, n, w, p, t
    filename = input("Please Insert File path of Config:")
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip('\n')
            data = line.split("=")
            if data[0] == "d ": #number of close points required
                d = int(data[1])
            elif data[0] == "n ": #number of sample points
                n = int(data[1])
            elif data[0] == "w ": 
                w = float(data[1])
            elif data[0] == "p ": #probability 
                p = float(data[1])
            elif data[0] == "t ": #threshold value
                t = float(data[1])
    print("Min. Number of close points required: ",d,"\n")
    print("Min. Number of Sample Points: ",n,"\n")
    print("w: ",w,"\n")
    print("Probability: ",p,"\n")
    print("Threshold value: ",t,"\n")
    
    
def fnd_inliners(points):
	global t
    #find projection matrix
	A = []
	for j in points:
		A.append([ origin_point[j][0], origin_point[j][1], origin_point[j][2], 1, 0, 0, 0, 0, -1 * image_point[j][0] * origin_point[j][0], -1 * image_point[j][0] * origin_point[j][1], -1 * image_point[j][0] * origin_point[j][2], -1 * image_point[j][0] * 1 ])
		A.append([ 0, 0, 0, 0, origin_point[j][0], origin_point[j][1], origin_point[j][2], 1, -1 * image_point[j][1] * origin_point[j][0], -1 * image_point[j][1] * origin_point[j][1], -1 * image_point[j][1] * origin_point[j][2], -1 * image_point[j][1] * 1 ])
	U, D, V = np.linalg.svd(A, full_matrices=True)
	M = np.split(V[11], 3)
	distances = []
	inliers = []
    #calculate distance and check wheter it is an inlier or outlier, if greater than threshold its outlier
	for l in range(0, num_lines):
		p = np.matmul(M, [ origin_point[l][0], origin_point[l][1], origin_point[l][2], 1])
		X_d = (image_point[l][0] - p[0]/p[2]) ** 2
		Y_d = (image_point[i][1] - p[1]/p[2]) ** 2
		dist = np.sqrt(X_d + Y_d)
		distances.append(dist)
		if dist < t:
			inliers.append(l)
	distances.sort()
	t = 1.5 * np.median(distances)
	return inliers



read_data()
read_config_parameter()
#find model
best_model_data = []
maxinl = 0
final_threshold = t
#find value of N i.e iteration
N = int(np.log(1 - p) / np.log(1 - w ** n))
for i in range(0, N):
	sample_points = []
    #take random sample points
	for i in range(0, n):
		sample_points.append(random.randint(0,267))
	inliers = fnd_inliners(sample_points)
    #check the no. of inliers found is greater than the min. of inliers we maintain
	if len(inliers) > d:
		recomputed_inliners = fnd_inliners(inliers)
		w = len(recomputed_inliners)/num_lines
		k = int(np.log(1 - p) / np.log(1 - w ** n))
		if len(recomputed_inliners) > maxinl:
			maxinl = len(recomputed_inliners)
			best_data = recomputed_inliners[:]
			final_threshold = t
print("Best Mode Details:\n")	
print("Number of inliers in data:", maxinl)
print("Best Model data: ", best_data)
print("Best threshold: ",final_threshold)
print("Number of iteration required: ", N)
    

