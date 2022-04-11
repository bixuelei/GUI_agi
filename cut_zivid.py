#import jinja2
from fileinput import filename
import numpy as np
import os
import torch.nn as nn
import pcl.pcl_visualization
from sklearn.cluster import DBSCAN
import open3d as o3d
########### used to transfer the rgb to r g b###########3
from struct import pack,unpack

cam_to_base_transform = [[ 6.3758686e-02 ,9.2318553e-01,-3.7902945e-01 ,4.5398907e+01],
 [ 9.8811066e-01,-5.1557920e-03 ,1.5365793e-01,-7.5876160e+02],
 [ 1.3990058e-01,-3.8432005e-01,-9.1253817e-01 ,9.6543054e+02],
 [ 0.0000000e+00 ,0.0000000e+00 ,0.0000000e+00 ,1.0000000e+00]]




def cut_motor(whole_scene):
    x_far=-360
    x_close=-230
    y_far=-940
    y_close=-640
    z_down=0
    z_up=250
    Corners = [(x_close,y_far,z_up), (x_close,y_close,z_up), (x_far,y_close,z_up), (x_far,y_far,z_up), (x_close,y_far,z_down), (x_close,y_close,z_down), (x_far,y_close,z_down), (x_far,y_far,z_down)]
    #Corners = [(35,880,300), (35,1150,300), (-150,1150,300), (-150,880,300), (35,880,50), (35,1150,50), (-150,1150,50), (-150,880,50)]
    cor_inCam = []
    for corner in Corners:
        cor_inCam_point = base_to_camera(np.array(corner))
        cor_inCam.append(np.squeeze(np.array(cor_inCam_point)))

    panel_1 = get_panel(cor_inCam[0], cor_inCam[1], cor_inCam[2])
    panel_2 = get_panel(cor_inCam[5], cor_inCam[6], cor_inCam[4])
    panel_3 = get_panel(cor_inCam[0], cor_inCam[3], cor_inCam[4])
    panel_4 = get_panel(cor_inCam[1], cor_inCam[2], cor_inCam[5])
    panel_5 = get_panel(cor_inCam[0], cor_inCam[1], cor_inCam[4])
    panel_6 = get_panel(cor_inCam[2], cor_inCam[3], cor_inCam[6])
    panel_list = {'panel_up':panel_1, 'panel_bot':panel_2, 'panel_front':panel_3, 'panel_behind':panel_4, 'panel_right':panel_5, 'panel_left':panel_6}

    patch_motor = []
    residual_scene=[]
    for point in whole_scene:
        point_cor = (point[0], point[1], point[2])
        if set_Boundingbox(panel_list, point_cor):
            patch_motor.append(point)
    return np.array(patch_motor)



def get_panel(point_1, point_2, point_3):

    x1 = point_1[0]
    y1 = point_1[1]
    z1 = point_1[2]

    x2 = point_2[0]
    y2 = point_2[1]
    z2 = point_2[2] 

    x3 = point_3[0]
    y3 = point_3[1]
    z3 = point_3[2]
    
    a = (y2-y1)*(z3-z1) - (y3-y1)*(z2-z1)
    b = (z2-z1)*(x3-x1) - (z3-z1)*(x2-x1)
    c = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    d = 0 - (a*x1 + b*y1 + c*z1)

    return (a, b, c, d)



def set_Boundingbox(panel_list, point_cor):

    if panel_list['panel_up'][0]*point_cor[0] + panel_list['panel_up'][1]*point_cor[1] + panel_list['panel_up'][2]*point_cor[2] + panel_list['panel_up'][3] <= 0 :   # panel 1
        if panel_list['panel_bot'][0]*point_cor[0] + panel_list['panel_bot'][1]*point_cor[1] + panel_list['panel_bot'][2]*point_cor[2] + panel_list['panel_bot'][3] >= 0 : # panel 2
            if panel_list['panel_front'][0]*point_cor[0] + panel_list['panel_front'][1]*point_cor[1] + panel_list['panel_front'][2]*point_cor[2] + panel_list['panel_front'][3] <= 0 : # panel 3
                if panel_list['panel_behind'][0]*point_cor[0] + panel_list['panel_behind'][1]*point_cor[1] + panel_list['panel_behind'][2]*point_cor[2] + panel_list['panel_behind'][3] >= 0 : # panel 4
                    if panel_list['panel_right'][0]*point_cor[0] + panel_list['panel_right'][1]*point_cor[1] + panel_list['panel_right'][2]*point_cor[2] + panel_list['panel_right'][3] >= 0 : #panel 5
                        if panel_list['panel_left'][0]*point_cor[0] + panel_list['panel_left'][1]*point_cor[1] + panel_list['panel_left'][2]*point_cor[2] + panel_list['panel_left'][3] >= 0 : # panel 6

                            return True
    return False



def base_to_camera(xyz, calc_angle=False):
    '''
    now do the base to camera transform
    '''

        # squeeze the first two dimensions
    xyz_transformed2 = xyz.reshape(-1, 3)  # [N=X*Y, 3]

        # homogeneous transformation
    if calc_angle:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.zeros((xyz_transformed2.shape[0], 1))))  # [N, 4]
    else:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.ones((xyz_transformed2.shape[0], 1))))  # [N, 4]

    cam_to_base_transform_ = np.matrix(cam_to_base_transform)
    base_to_cam_transform = cam_to_base_transform_.I
    xyz_transformed2 = np.matmul(base_to_cam_transform, xyz_transformed2.T).T  # [N, 4]

    return xyz_transformed2[:, :-1].reshape(xyz.shape)  # [X, Y, 3]



def camera_to_base(xyz, calc_angle=False):
    '''
    '''
        # squeeze the first two dimensions
    xyz_transformed2 = xyz.reshape(-1, 3)  # [N=X*Y, 3]

        # homogeneous transformation
    if calc_angle:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.zeros((xyz_transformed2.shape[0], 1))))  # [N, 4]
    else:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.ones((xyz_transformed2.shape[0], 1))))  # [N, 4]


    xyz_transformed2 = np.matmul(cam_to_base_transform, xyz_transformed2.T).T  # [N, 4]

    return xyz_transformed2[:, :-1].reshape(xyz.shape)  # [X, Y, 3]




def open3d_save_pcd(pc,filename):
    sampled = np.asarray(pc)
    PointCloud_koordinate = sampled[:, 0:3]

    #visuell the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    o3d.io.write_point_cloud(filename, point_cloud, write_ascii=True)


if __name__ == "__main__":
    file_path = "/home/bi/study/thesis/data/current_finetune/pcdfile"
    save_path = "/home/bi/study/thesis/data/current_finetune/cut_pcdfile"
    List_zivid = os.listdir(file_path)
    List_zivid.sort()
    for motor_scene in List_zivid:
        cloud = pcl.load_XYZRGB(file_path+'/'+motor_scene)
        num_points=cloud.size
        points=[]
        for i in range(num_points):
            inter=unpack('i',pack('f',cloud[i][3]))[0]
            r=(inter>>16) & 0x0000ff
            g=(inter>>8) & 0x0000ff
            b=(inter>>0) & 0x0000ff
            points.append([cloud[i][0], cloud[i][1], cloud[i][2],r,g,b])
        points=np.array(points)
        points=cut_motor(points)
        #save the pcd file
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
        point_cloud.colors = o3d.utility.Vector3dVector(points[:, 3:6]/255)
        o3d.io.write_point_cloud(save_path+'/'+motor_scene, point_cloud)
        # o3d.visualization.draw_geometries([point_cloud])
        print("one get cut")
