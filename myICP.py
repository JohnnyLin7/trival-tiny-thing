import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadData(fileName):
    fin = open(fileName, "r")
    lines = fin.readlines()
    cnt = len(lines)
    data = np.zeros([cnt,3])
    for i in range(2,cnt):
        data[i - 2] = lines[i].split(' ')
    fin.close()
    return data

def outputData(fileName, data):
    fout = open(fileName, "w")
    for i in range(data.shape[0]):
        fout.write(str(data[i][0]) + ',' + str(data[i][1]) + ',' + str(data[i][2]) + '\n')
    fout.close()

def nearest_point(P, Q):
    P = np.array(P)
    Q = np.array(Q)
    dis = np.zeros(P.shape[0])
    index = np.zeros(Q.shape[0], dtype = int)

    for i in range(P.shape[0]):
        minDis = np.inf
        for j in range(Q.shape[0]):
            tmp = np.linalg.norm(P[i] - Q[j], ord = 2)
            if minDis > tmp:
                minDis = tmp
                index[i] = j
        dis[i] = minDis
    return dis, index

def find_optimal_transform(P, Q):
    meanP = np.mean(P, axis = 0)
    meanQ = np.mean(Q, axis = 0)
    P_ = P - meanP
    Q_ = Q - meanQ

    W = np.dot(Q_.T, P_)
    U, S, VT = np.linalg.svd(W)
    R = np.dot(U, VT)
    if np.linalg.det(R) < 0:
       R[2, :] *= -1

    T = meanQ.T - np.dot(R, meanP.T)
    return R, T

def icp(src, dst, maxIteration=50, tolerance=0.001, controlPoints=100):
    A = np.array(src)
    B = np.array(dst)
    lastErr = 0
    if (A.shape[0] != B.shape[0]):
        length = min(A.shape[0], B.shape[0])
        length = min(length, controlPoints)
        sampleA = random.sample(range(A.shape[0]), length)
        sampleB = random.sample(range(B.shape[0]), length)
        P = np.array([A[i] for i in sampleA])
        Q = np.array([B[i] for i in sampleB])
    else:
        length = A.shape[0]
        if (length > controlPoints):
            sampleA = random.sample(range(A.shape[0]), length)
            sampleB = random.sample(range(B.shape[0]), length)
            P = np.array([A[i] for i in sampleA])
            Q = np.array([B[i] for i in sampleB])
        else :
            P = A
            Q = B

    for i in range(maxIteration):
        print("Iteration : " + str(i) + " with Err : " + str(lastErr))
        dis, index = nearest_point(P, Q)
        R, T = find_optimal_transform(P, Q[index,:])
        A = np.dot(R, A.T).T + np.array([T for j in range(A.shape[0])])
        P = np.dot(R, P.T).T + np.array([T for j in range(P.shape[0])])

        meanErr = np.sum(dis) / dis.shape[0]
        if abs(lastErr - meanErr) < tolerance:
            break
        lastErr = meanErr


    R, T = find_optimal_transform(A, np.array(src))
    return R, T, A


def downsample_point_cloud(data, ratio):
    """
    Downsample a point cloud by keeping only a fraction of the points.
    """
    num_points = len(data)
    num_points_to_keep = int(num_points * ratio)
    indices_to_keep = random.sample(range(num_points), num_points_to_keep)
    return data[indices_to_keep]



def plot_point_clouds(original, registered, downsample_ratio=0.1):
    # Downsample the point clouds
    original_downsampled = downsample_point_cloud(original, downsample_ratio)
    registered_downsampled = downsample_point_cloud(registered, downsample_ratio)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(original_downsampled[:, 0], original_downsampled[:, 1], original_downsampled[:, 2], c='b', marker='o', label='Original')
    ax.scatter(registered_downsampled[:, 0], registered_downsampled[:, 1], registered_downsampled[:, 2], c='r', marker='x', label='Registered')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()


if __name__ == "__main__":

    data1 = loadData("9_A.txt")
    data2 = loadData("9_B.txt")

    _, _, data2_ = icp(data2, data1, maxIteration = 100, tolerance = 0.00001, controlPoints = 1000)
    outputData("result.txt", data2_)
    # Visualize the original and registered point clouds
    plot_point_clouds(data1, data2_)
