from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import numpy as np
import random
from numpy.linalg import matrix_rank
import math
import cv2
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd

frames = []

path = "stereo/centre/"

for frame in os.listdir(path):
    frames.append(frame)
    frames.sort()

fx, fy, c_x, c_y, camera_img, LUT = ReadCameraModel('model/')
K = np.array([[fx , 0 , c_x],[0 , fy , c_y],[0 , 0 , 1]])

def CamPosematrix(E_matrix):
    u, s, v = np.linalg.svd(E_matrix, full_matrices=True)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    c1 = u[:, 2]
    r1 = u @ w @ v
    
    if np.linalg.det(r1) < 0:
        c1 = -c1 
        r1 = -r1
    c1 = c1.reshape((3,1))
    
    c2 = -u[:, 2]
    r2 = u @ w @ v
    if np.linalg.det(r2) < 0:
        c2 = -c2 
        r2 = -r2 
    c2 = c2.reshape((3,1))
    
    c3 = u[:, 2]
    r3 = u @ w.T @ v
    if np.linalg.det(r3) < 0:
        c3 = -c3 
        r3 = -r3 
    c3 = c3.reshape((3,1)) 
    
    c4 = -u[:, 2]
    r4 = u @ w.T @ v
    if np.linalg.det(r4) < 0:
        c4 = -c4 
        r4 = -r4 
    c4 = c4.reshape((3,1))
    
    return [r1, r2, r3, r4], [c1, c2, c3, c4]

def obtainEulerAngles(rot_mat) :
    eu1 = math.sqrt(rot_mat[0,0] * rot_mat[0,0] +  rot_mat[1,0] * rot_mat[1,0])
    singular_val = eu1 < 1e-6
 
    if  not singular_val :
        x = math.atan2(rot_mat[2,1] , rot_mat[2,2])
        y = math.atan2(-rot_mat[2,0], eu1)
        z = math.atan2(rot_mat[1,0], rot_mat[0,0])

    else :
        x = math.atan2(-rot_mat[1,2], rot_mat[1,1])
        y = math.atan2(-rot_mat[2,0], eu1)
        z = 0
    return np.array([x*180/math.pi, y*180/math.pi, z*180/math.pi])

def fundamentalMatrix(edge1, edge2): 
    A_x = np.empty((8, 9))

    for i in range(0, len(edge1)):
        x_1 = edge1[i][0]
        y_1 = edge1[i][1]
        x_2 = edge2[i][0]
        y_2 = edge2[i][1]
        A_x[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

    u, s, v = np.linalg.svd(A_x, full_matrices=True)  
    f = v[-1].reshape(3,3)
    u1,s1,v1 = np.linalg.svd(f) 
    s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]])
    F = u1 @ s2 @ v1    
    return F  

def FmatrixCond(x1,x2,F): 
    x11=np.array([x1[0],x1[1],1]).T
    x22=np.array([x2[0],x2[1],1])
    return abs(np.squeeze(np.matmul((np.matmul(x22,F)),x11)))

def Essentialmatrix(calibrationMatrix, Fmatrix):
    tempM = np.matmul(np.matmul(calibrationMatrix.T, Fmatrix), calibrationMatrix)
    u, s, v = np.linalg.svd(tempM, full_matrices=True)
    S_F = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    flag = np.matmul(u, S_F)
    E_matrix = np.matmul(flag, v)
    return E_matrix

def Homogenousmatrix(rot_mat, t):
    i = np.column_stack((rot_mat, t))
    a = np.array([0, 0, 0, 1])
    H = np.vstack((i, a))
    return H

def getTriangulationPoint(m1, m2, point1, point2):
    oldx = np.array([[0, -1, point1[1]], [1, 0, -point1[0]], [-point1[1], point1[0], 0]])
    oldxdash = np.array([[0, -1, point2[1]], [1, 0, -point2[0]], [-point2[1], point2[0], 0]])
    A1 = oldx @ m1[0:3, :] 
    A2 = oldxdash @ m2
    A_x = np.vstack((A1, A2))
    u, s, v = np.linalg.svd(A_x)
    new1X = v[-1]
    new1X = new1X/new1X[3]
    new1X = new1X.reshape((4,1))
    return new1X[0:3].reshape((3,1))

def disambiguiousPose(RotationMatrix, CameraCenter, features1, features2):
    check = 0
    Horigin = np.identity(4)
    for index in range(0, len(RotationMatrix)):
        angles = obtainEulerAngles(RotationMatrix[index])
        if angles[0] < 50 and angles[0] > -50 and angles[2] < 50 and angles[2] > -50:
            count = 0
            newP = np.hstack((RotationMatrix[index], CameraCenter[index]))
            for i in range(0, len(features1)):
                temp1x = getTriangulationPoint(Horigin[0:3,:], newP, features1[i], features2[i])
                thirdrow = RotationMatrix[index][2,:].reshape((1,3))
                if np.squeeze(thirdrow @ (temp1x - CameraCenter[index])) > 0: 
                    count = count + 1
            if count > check:
                check = count
                Translation_final = CameraCenter[index]
                Rotation_final = RotationMatrix[index]
                
    if Translation_final[2] > 0:
        Translation_final = -Translation_final
                
    return Rotation_final, Translation_final
    
H_Start = np.identity(4)
p_0 = np.array([[0, 0, 0, 1]]).T
flag = 0

data_points = []
for index in range(19, len(frames)-1):
    print(frames[index], index)
    img1 = cv2.imread("stereo/centre/" + str(frames[index]), 0)
    colorimage1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
    undistortedimage1 = UndistortImage(colorimage1,LUT)  
    gray1 = cv2.cvtColor(undistortedimage1,cv2.COLOR_BGR2GRAY)
    
    img2 = cv2.imread("stereo/centre/" + str(frames[index + 1]), 0)
    colorimage2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
    undistortedimage2 = UndistortImage(colorimage2,LUT)  
    gray2 = cv2.cvtColor(undistortedimage2,cv2.COLOR_BGR2GRAY)

    grayImage1 = gray1[200:650, 0:1280]
    grayImage2 = gray2[200:650, 0:1280]

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(grayImage1,None)
    kp2, des2 = sift.detectAndCompute(grayImage2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    features1 = []
    features2 = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            features1.append(kp1[m.queryIdx].pt)
            features2.append(kp2[m.trainIdx].pt)
        
    Total_inliers = 0
    FinalFundamentalMatrix = np.zeros((3,3))
    inlier1 = []
    inlier2 = []
    for i in range(0, 50):
        count = 0
        Extracted_points = []
        Frame1_features = []
        Frame2_features = []
        TemporaryFeatures_1 = []
        TemporaryFeatures_2 = []
        
        while(True):
            num = random.randint(0, len(features1)-1)
            if num not in Extracted_points:
                Extracted_points.append(num)
            if len(Extracted_points) == 8:
                break

        for point in Extracted_points:
            Frame1_features.append([features1[point][0], features1[point][1]])
            Frame2_features.append([features2[point][0], features2[point][1]])
    
        FundMatrix = fundamentalMatrix(Frame1_features, Frame2_features)

        for number in range(0, len(features1)):
            if FmatrixCond(features1[number], features2[number], FundMatrix) < 0.01:
                count = count + 1
                TemporaryFeatures_1.append(features1[number])
                TemporaryFeatures_2.append(features2[number])

        if count > Total_inliers:
            Total_inliers = count
            FinalFundamentalMatrix = FundMatrix
            inlier1 = TemporaryFeatures_1
            inlier2 = TemporaryFeatures_2
    
    E_matrix = Essentialmatrix(K, FinalFundamentalMatrix)

    RotationMatrix, Tlist = CamPosematrix(E_matrix)
    rot_mat, T = disambiguiousPose(RotationMatrix, Tlist, inlier1, inlier2)

    H_Start = H_Start @ Homogenousmatrix(rot_mat, T)
    p_projection = H_Start @ p_0

    print('x- ', p_projection[0])
    print('y- ', p_projection[2])
    data_points.append([p_projection[0][0], -p_projection[2][0]])
    plt.scatter(p_projection[0][0], -p_projection[2][0], color='r')
    
    if cv2.waitKey(0) == 27:
        break
    flag = flag + 1

cv2.destroyAllWindows()
df = pd.DataFrame(data_points, columns = ['X', 'Y'])
df.to_excel('coordinates.xlsx')
plt.show()

