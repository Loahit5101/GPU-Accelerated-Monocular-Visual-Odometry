import os
import numpy as np
import cv2
import argparse
from glob import glob
from pose_evaluation_utils import *

width = 1241.0
height = 376.0
fx, fy, cx, cy = [718.8560, 718.8560, 607.1928, 185.2157]

seq = '00'
gt_pose_dir = '/home/loahit/Downloads/KITTI_odometry/data_odometry_poses/dataset/poses/09.txt'
img_data_dir = '/home/loahit/Downloads/KITTI_odometry/data_odometry_gray/dataset/sequences/09/image_0'
if __name__ == "__main__":

    trajMap = np.zeros((2000,2000, 3), dtype=np.uint8)
  
    img_list = glob(img_data_dir + '/*.png')
    img_list.sort()
    num_frames = len(img_list)
    print(num_frames)
   
    with open(gt_pose_dir) as f:
         gt_pose_str_list = f.readlines()

    for i in range(num_frames):

        
        curr_img = cv2.imread(img_list[i])
        if i == 0:
            curr_R = np.eye(3)
            curr_t = np.array([0, 0, 0])
        else:

            prev_img = cv2.imread(img_list[i-1], 0)

            orb = cv2.ORB_create(nfeatures=6000)
            kp1, des1 = orb.detectAndCompute(prev_img, None)
            kp2, des2 = orb.detectAndCompute(curr_img, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            img_matching = cv2.drawMatches(prev_img, kp1, curr_img, kp2, matches[0:100], None)
            cv2.imshow('feature matching', img_matching)

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1)
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))

            if i == 1:
                curr_R = R
                curr_t = t
            else:
                curr_R = np.matmul(prev_R, R)
                curr_t = np.matmul(prev_R, t) + prev_t

            curr_img_kp = cv2.drawKeypoints(curr_img, kp2, None, color=(0, 255, 0), flags=0)
            cv2.imshow('keypoints from current image', curr_img_kp)

        prev_R = curr_R
        prev_t = curr_t

        offset_draw = (int(2000/2))
        cv2.circle(trajMap, (int(curr_t[0])+offset_draw, int(curr_t[2])+offset_draw), 1, (255,0,0), 2)
        cv2.imshow('Trajectory', trajMap)
        cv2.waitKey(1)

    cv2.imwrite('trajMap.png', trajMap)

