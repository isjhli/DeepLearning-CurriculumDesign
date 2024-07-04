import cv2
import numpy as np

# -----------------------------------双目相机的基本参数---------------------------------------------------------
#   left_camera_matrix          左相机的内参矩阵
#   right_camera_matrix         右相机的内参矩阵
#
#   left_distortion             左相机的畸变系数    格式(K1,K2,P1,P2,0)
#   right_distortion            右相机的畸变系数
# -------------------------------------------------------------------------------------------------------------
# 左镜头的内参，如焦距
left_camera_matrix = np.array([[523.3686, -1.7433, 272.3166],
                               [0, 524.3238, 249.7574],
                               [0, 0, 1.0000]])
right_camera_matrix = np.array([[521.7087, -1.8426, 264.4720],
                                [0, 523.5482, 253.6896],
                                [0, 0, 1.0000]])

# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
left_distortion = np.array([[-0.0455, 0.2672, -0.0058, -0.0107, -0.4019]])
right_distortion = np.array([[-0.0106, 0.0996, -0.0025, -0.0106, -0.1911]])

# 旋转矩阵
R = np.array([[1.0000, 0.0002, 0.0050],
              [-0.0002, 1.0000, -0.0094],
              [-0.0050, 0.0094, 0.9999]])
# 平移矩阵
T = np.array([-58.4786, -0.6207, -0.5327])
size = (640, 480)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
# print(Q)
