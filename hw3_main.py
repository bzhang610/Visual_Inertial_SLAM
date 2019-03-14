import numpy as np
from utils import *


if __name__ == '__main__':
    filename = "./data/0042.npz"
    t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

    # (a) IMU Localization via EKF Prediction
    iTw, wTi = imu_predict(t,linear_velocity,rotational_velocity,10e-7)
    # (b) Landmark Mapping via EKF Update
    landmarks = visual_mapping(features,iTw, K,b,cam_T_imu,1000)
    # (c) Visual-Inertial SLAM (Extra Credit)
    slam_iTw, slam_wTi,slam_landmarks = vi_slam(t,linear_velocity,rotational_velocity,features, K,b,cam_T_imu,10e-7,1000)
    # You can use the function below to visualize the robot pose over time
    #visualize_trajectory_2d(world_T_imu,show_ori=True)
    show_result(wTi,landmarks,'Seperate',show_ori=True)
    show_result(slam_wTi,slam_landmarks,'SLAM',show_ori=True)