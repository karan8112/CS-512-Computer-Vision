import cv2
import numpy as np
from as6_function import normalise,F_matrix, epipolar


norm_left = []
norm_right = []
left_epipole = 0
right_epipole = 0
ix = 0
iy = 0


#get x,y coordinates of onclick of mouse event
def on_click_mouse(event, x, y, flags, params):
   global ix,iy  
   if event==cv2.EVENT_LBUTTONDOWN:
       ix,iy=x,y


def left_image_point(left):
    total_left_point=[]
    count_left = 0
    while(1):
        cv2.imshow('left image frame',left)
        cv2.setMouseCallback('left image frame',on_click_mouse, 0)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('l'):
            count_left = count_left +1
            cv2.putText(left,str(count_left),(ix,iy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
            xy_left = []
            xy_left.append(ix)
            xy_left.append(iy)
            xy_left.append(1)
            total_left_point.append(xy_left)
    return total_left_point
        
    
       
       
def right_image_point(right):
    total_right_point=[]
    count_right = 0
    while(1):
        cv2.imshow('right image frame',right)
        cv2.setMouseCallback('right image frame',on_click_mouse, 0)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('r'):
            count_right = count_right +1
            cv2.putText(right,str(count_right),(ix,iy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
            xy_right = []
            xy_right.append(ix)
            xy_right.append(iy)
            xy_right.append(1)
            total_right_point.append(xy_right)
    return total_right_point
        
        


def press_mouse(event, x, y, flags, params):
   global ix,iy  
   if event==cv2.EVENT_LBUTTONDOWN:
       ix,iy=x,y
       
       
def left_right_epipolarline(left, right, F, rows_r, cols_r):
    while(1):
        cv2.imshow('left image frame',left)
        cv2.setMouseCallback('left image frame',on_click_mouse, 0)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
        right[int(right_epipole[1]),int(right_epipole[0])] = [0,255,255]
        line = np.array([ix,iy,1])
        d = np.matmul(line,F.T)
        a = d[0]
        b = d[1]
        c = d[2]
        if a >= b:
            for x in range(0,rows_r):
                y = int((-c -a*x)/b)
                if y < right.shape[1] and y > 0:
                    right[y,x] = [0,0,255] 
        else:
            for y in range(0,cols_r):
                x = int((-c -b*y)/a)
                if x < right.shape[0] and x > 0:
                    right[y,x] = [0,255,0]
        cv2.imshow('right image frame',right)
        
        
def right_left_epipolarline(left, right, F, rows_r, cols_r):
    while(1):
        cv2.imshow('right image frame',right)
        cv2.setMouseCallback('right image frame',on_click_mouse, 0)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
        left[int(left_epipole[1]),int(left_epipole[0])] = [0,255,255]
        line = np.array([ix,iy,1])
        d = np.matmul(line,F.T)
        a = d[0]
        b = d[1]
        c = d[2]
        if a >= b:
            for x in range(0,rows_r):
                y = int((-c -a*x)/b)
                if y < left.shape[1] and y > 0:
                    left[y,x] = [0,0,255] 
        else:
            for y in range(0,cols_r):
                x = int((-c -b*y)/a)
                if x < left.shape[0] and x > 0:
                    left[y,x] = [0,255,0]
        cv2.imshow('left image frame',left)
    cv2.destroyAllWindows()
    
        
    
    
def main():
    #left_pt = []
    #right_pt = []
    global left_epipole, right_epipole
    k = input("Press h for help function to know how to run the program else press c:")
    if(k == 'h'):
        print("This Program performs Epipolar Estimation by calculating the Fundamental Matrix from corresponding points of two images. \n Run the as6.py file using python then input 'c' for run the program.\na) First you have to run the as6.py file using the python.\nb) Now it will ask to enter the input, if you want to see the help function enter ‘h’ else enter ‘c’ to run the program.\nc) Now you have to pass the path of both left and right image.\nd) Then the first left image will pop up, now you have to select at least 8 points by pressing left and pressing ‘l’ button of keyboard to select the corresponding points. After selecting all the points press esc key.\ne) Then the right image will pop up, now you have to select at least 8 points by pressing left and pressing ‘r’ button of keyboard to select the corresponding points. After selecting all the points press esc key.\nf) After selecting all points, Fundamental Matrix and epipole of both left and right image will display.\ng) Now to mark epipolar line, select the point in left image and it will a draw a corresponding epipolar line on right image. Now press esc to do the same to draw the epipolar line on left image for right image points.")
    if(k == 'c'):
        left_image = input("Enter the path of the left file image:")
        left = cv2.imread(left_image)
        right_image = input("Enter the path of the right file image:")
        right = cv2.imread(right_image)
    
        row_r,col_r,_ = right.shape
        row_l,col_l,_ = left.shape
    
    
        #specify the corresponding points of left image
        left_pt = left_image_point(left)
        print(left_pt)
        #specify the corresponding points of left image
        right_pt = right_image_point(right)
        print(right_pt)
    
        #now normalize the points of left and right image one by one calling normalization function
        M_left = normalise(left_pt)
        for i in left_pt:
            new= np.matmul(M_left,i)
            norm_left.append(new)
        M_right = normalise(right_pt)
        for i in right_pt:
            new2 =np.matmul(M_right,i)
            norm_right.append(new2)
    
        #calculate f matrix
        F = F_matrix(norm_left,norm_right)
        F = np.dot(M_left.T,np.dot(F,M_right))
        print("\nFundamental Matrix :\n ",F)
    
        #calculate epipolar points
        e_l,e_r=epipolar(F)
        t,y,u =e_l
        left_epipole = [t/u,y/u]
        i,o,p = e_r
        right_epipole = [i/p,o/p]
        print("left epipole point:\n",left_epipole)
        print("right epipole point\n:",right_epipole)
    
        #plot epipolar lines of left point on right image
        left_right_epipolarline(left, right, F, row_r, col_r)
        #plot epipolar lines of right point on lef image
        right_left_epipolarline(left, right, F, row_r, col_r)
    



if __name__ == '__main__':
    main()