import argparse
import open3d as o3d
import pandas as pd
import sys
import json
import cv2
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys, os
import time
import struct
from PIL import Image

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

rot_180x = np.array([[1, 0, 0,0], [0, -1, 0,0], [0, 0, -1,0],  [0, 0, 0, 1]])
def visualize_point_cloud(ply_file_path):
    # Load the point cloud from the .ply file
    point_cloud = o3d.io.read_point_cloud(ply_file_path)
    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])


def generate_intrinsic(camera):
    camera_params = o3d.camera.PinholeCameraParameters()
    # import pdb; pdb.set_trace()

    width = camera['width']
    height = camera['height']
    f_x = camera['focal_x']
    f_y = camera['focal_y']
    c_x = camera['c_x']
    c_y = camera['c_y']
    # camera_params.intrinsic.set_intrinsics(width,height,f_x, f_y, c_x, c_y)
    int_matrix = np.array([[f_x*max(width,height), 0, 1/2*(width-1)],
                        [0, f_x*max(width,height), 1/2*(height-1)],
                        [0,   0,   1]])
    return 1, int_matrix

def most_repeated_row(lst):
    # Convert the list to a NumPy array
    arr = np.array(lst)
    
    # Filter out rows equal to [0, 0, 0]
    filtered_arr = arr[~np.all(arr == [0, 0, 0], axis=1)]
    if len(filtered_arr)==0:
        return [0,0,0]
    
    # Count occurrences of each row
    unique_rows, counts = np.unique(filtered_arr, axis=0, return_counts=True)
    
    # Find the index of the most repeated row
    max_count_idx = np.argmax(counts)
    
    # Get the most repeated row
    most_common_row = unique_rows[max_count_idx]
    
    return most_common_row

def rotate_pointcloud(nube, rot):
    nube_aux = []
    for i in nube.points:
        nube_aux.append(np.dot(i, rot))

    nube_np = np.asarray(nube_aux)
    nube_def = o3d.geometry.PointCloud()
    nube_def.points = o3d.utility.Vector3dVector(nube_np)

    return nube_def


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def extract_filenames(nvm_filepath):
    filenames = []

    with open(nvm_filepath, 'r') as file:
        for line in file:
            # Check if the line contains an image reference
            if line.startswith('images/'):
                # Split the line to get the filename
                parts = line.split()
                filename_with_extension = parts[0]
                
                # Extract the filename without the .tif extension
                filename = filename_with_extension.split('.tif')[0] 
                
                # Append the filename to the list
                filenames.append(filename)
    return filenames

class HeaderDepthDataRaw:
    HAS_DEPTH = 1 << 0
    HAS_NORMAL = 1 << 1
    HAS_CONF = 1 << 2

    def __init__(self):
        self.name = 0
        self.type = 0
        self.padding = 0
        self.imageWidth = 0
        self.imageHeight = 0
        self.depthWidth = 0
        self.depthHeight = 0
        self.dMin = 0.0
        self.dMax = 0.0

def read_depth(path):
    with open(path, "rb") as f:
        header_data = f.read(struct.calcsize("=HBBIIIIff"))
        header = HeaderDepthDataRaw()
        header.name, header.type, header.padding, header.imageWidth, header.imageHeight, header.depthWidth, header.depthHeight, header.dMin, header.dMax = struct.unpack("=HBBIIIIff", header_data)
        # import pdb; pdb.set_trace()
        nFileNameSize = struct.unpack("=H", f.read(2))[0]
        # print(nFileNameSize)
        
        imageFileName = f.read(nFileNameSize).decode("utf-8")
        # print(imageFileName)
        
        found = imageFileName.find("frame")
        short_name = imageFileName[found:found+11]
        short_name = short_name + "png"
        # print(short_name)
        
        depthMap = []
        for _ in range(header.depthHeight):
            row = struct.unpack("={}f".format(header.depthWidth), f.read(4 * header.depthWidth))
            depthMap.append(row[50:] + row[:50])

        
    # print(len(depthMap))
    depth_image = Image.new("L", (header.depthWidth, header.depthHeight))
    depth_image.putdata([item for sublist in ( depthMap) for item in sublist])
    return depth_image

def compute_depth(K, R, T, P_world, p_image):
    # Ensure input is in correct shape
    P_world = np.asarray(P_world).reshape(-1, 3)
    p_image = np.asarray(p_image).reshape(-1, 2)
    # import pdb; pdb.set_trace()
    
    # Transform 3D world points to camera coordinates
    P_camera = (R @ P_world.T).T + T.T  # Resulting shape: (n, 3)
    
    # Extract X, Y, Z coordinates in the camera frame
    X_c, Y_c, Z_c = P_camera[:, 0], P_camera[:, 1], P_camera[:, 2]  # Each is shape (n,)
    
    # Use intrinsic parameters to project points to the image plane
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Given 2D image coordinates
    u, v = p_image[:, 0], p_image[:, 1]  # Each is shape (n,)
    # import pdb; pdb.set_trace()

    
    # Calculate depth using the projection equations
    Z_c_computed_u = np.asarray(fx * X_c).reshape(-1,) / np.asarray(u - cx)  # Shape: (n,)
    Z_c_computed_v = np.asarray(fy * Y_c).reshape(-1,) / (v - cy)  # Shape: (n,)
    
    # Take the average depth from u and v computations
    Z_c_computed = (Z_c_computed_u + Z_c_computed_v) / 2  # Shape: (n,)
    
    return Z_c_computed


def get_depth_values(depth_image, uv_points):
    """
    Get the depth values from the depth image at the specified (u, v) points.
    
    :param depth_image: 2D numpy array representing the depth image.
    :param uv_points: 2D numpy array of shape (n, 2) containing (u, v) points.
    :return: 1D numpy array of depth values at the specified points.
    """
    # Ensure uv_points is a numpy array with shape (n, 2)
    uv_points = np.asarray(uv_points).reshape(-1, 2)
    
    # Extract u and v coordinates (Note: u is column index, v is row index)
    u = uv_points[:, 0]
    v = uv_points[:, 1]
    
    # Ensure u and v are within the bounds of the depth image
    u = np.clip(u, 0, depth_image.shape[1] - 1)
    v = np.clip(v, 0, depth_image.shape[0] - 1)
    
    # Get depth values at the specified points
    depth_values = depth_image[v, u]
    
    return depth_values



def read_camera(path_to_cameras, ply_file_path, output_path, path_to_project, path_to_segmentation, visualize = False):


    # Path to your .nvm file
    nvm_filepath = '{}/opensfm/undistorted/reconstruction.nvm'.format(path_to_project)
    # Get the ordered list of filenames
    ordered_filenames = extract_filenames(nvm_filepath)

    point_cloud = o3d.io.read_point_cloud(ply_file_path) 
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=0.2)  # Adjust voxel size and downsample the points
    points = np.asarray(downsampled_point_cloud.points)
    o3d.visualization.draw_geometries([downsampled_point_cloud])

    print("###################### INFO ######################")
    print("Original point-cloud size : ", (np.asarray(point_cloud.points)).shape )
    print("Down-sampled point-cloud size : ", points.shape )

    #read camera parameters
    reconstruction = pd.read_json(path_to_cameras)
    shots = reconstruction['shots'].values[0]
    id = [key for key in reconstruction['cameras'].values[0].keys()]
    camera = reconstruction['cameras'].values[0][id[0]]

    start_time = time.perf_counter()

    # get camera intrinsic matrix 
    _ , int_matrix = generate_intrinsic(camera)
    images = list(shots.keys())
    point_cloud_color_list = []
    for i,img_name in tqdm(enumerate(sorted(images))):

        # reduce number of images
        if i%2==1:
            continue

        # Load the image
        # segmentation image
        image_segmentation = cv2.imread("{}/{}".format(path_to_segmentation, img_name))  # Replace "your_image_path.jpg" with the path to your image file
        image_segmentation = cv2.cvtColor(image_segmentation, cv2.COLOR_BGR2RGB)
        # RGB image
        image_rgb =  cv2.cvtColor(cv2.imread("{}/images/{}".format(path_to_project, img_name)),cv2.COLOR_BGR2RGB)  # Replace "your_image_path.jpg" with the path to your image file
        # depth image
        image_depth = read_depth("{}/opensfm/undistorted/openmvs/depthmaps/depth{:04d}.dmap".format(path_to_project, ordered_filenames.index('images/{}'.format(img_name))))  # Replace "your_image_path.jpg" with the path to your image file
        resized_image = image_depth.resize((image_rgb.shape[1],image_rgb.shape[0]), Image.Resampling.NEAREST)

        # generate extrinsic camera metrics 
        rotation = shots[img_name]['rotation']
        rotation_matrix = cv2.Rodrigues(np.matrix(rotation))[0]
        translation_matrix = np.matrix(shots[img_name]['translation']).T
        RT_matrix = np.append(rotation_matrix,translation_matrix,1)
        RT_matrix4x4 = np.vstack((RT_matrix,[0,0,0,1]))
        # distortion = np.array([camera['k1'], camera['k2'], 0., 0.])
        distortion = np.array([0, 0, 0., 0.])


        # project ply to the image plane 
        pixels, _ = cv2.projectPoints(np.asarray(downsampled_point_cloud.points), rotation_matrix, translation_matrix, int_matrix, distortion)
        points_2d = pixels[:,0,:] # just changing the dimention 
        # index of the points fall in the image plane
        mask = (points_2d[:, 0] > 0) & (points_2d[:, 1] > 0) & (points_2d[:, 0] < camera['width']) & (points_2d[:, 1] < camera['height']) 
        indices = np.where(mask)[0]
        points_2D = points_2d[:,:2]
        masked_array = points_2D[indices].astype(int)

        # color the points projected on to image plane based on the semantic segmentation image 
        semantic_values = image_segmentation[np.asarray(masked_array[:, 1].astype(int)).reshape(-1,), np.asarray(masked_array[:, 0].astype(int)).reshape(-1,)]/255
        semanticly_colored_points = np.zeros(np.array(downsampled_point_cloud.colors).shape)
        semanticly_colored_points[indices] = semantic_values
        

        # clean the projected points based on depth values
        all_points_2D, _ = cv2.projectPoints(np.asarray(downsampled_point_cloud.points), rotation_matrix, translation_matrix, int_matrix, distortion)
        all_points_2D_reshaped = all_points_2D[:,0,:]
        depth_computed = compute_depth(int_matrix, rotation_matrix, translation_matrix, np.asarray(downsampled_point_cloud.points), all_points_2D_reshaped)
        depth_gt_values = get_depth_values((np.asarray(resized_image)!=255)*np.asarray(resized_image), all_points_2D_reshaped.astype(int))
        depth_not_passed_points_idx = np.where((depth_computed -depth_gt_values)>0.8)
        semanticly_colored_points[depth_not_passed_points_idx] = np.array([0,0,0])
        point_cloud_color_list.append(semanticly_colored_points)


        if visualize :
            fig = plt.figure(figsize=( 40, 20), dpi=100)

            # visualize RGB, seg, depth images
            ax = fig.add_subplot(1, 3, 1)
            ax.imshow(image_rgb)

            ax1 = fig.add_subplot(1, 3, 2)
            ax1.imshow(((np.asarray(resized_image)!=255)*np.asarray(resized_image)), cmap='viridis')

            ax2 = fig.add_subplot(1, 3, 3)
            ax2.imshow(image_segmentation)

            plt.show()
            plt.clf()

    # Create a voxel grid geometry of the semantic ply
    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(np.array(downsampled_point_cloud.points))
    points_semantics = np.zeros(np.array(downsampled_point_cloud.colors).shape)

    # choose the most repeted color for each point of the pointcloud
    for index in tqdm(range(np.array(downsampled_point_cloud.points).shape[0])):
        points_semantics[index] = most_repeated_row(np.asarray(point_cloud_color_list)[:,index,:])
    pcd_final.colors = o3d.utility.Vector3dVector(points_semantics)

    end_time = time.perf_counter()
    program_time = (end_time - start_time)/ len(images)
    print("The number of used images : " , int(len(images)/2))
    print("The program running time : " , program_time)

    # voxel representation
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_final, voxel_size=0.2)
    o3d.io.write_voxel_grid("{}/semantic_segmentation_pointcloud.ply".format(output_path), voxel_grid)  # Change the file format and filename as needed
    o3d.visualization.draw_geometries([voxel_grid])



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Visualize a 3D point cloud from a .ply file")
    parser.add_argument("ply_file", type=str, help="Path to the .ply file")
    parser.add_argument("path_to_cameras", type=str, help="Path to the camera informations file")
    parser.add_argument("path_to_segmentation", type=str, help="Path to the camera informations file")
    parser.add_argument("path_to_project", type=str, help="Path to the output folder file")
    parser.add_argument("output_path", type=str, help="Path to the output folder file")
    # parser.add_argument("visualize", type=bool, default= False)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    read_camera(args.path_to_cameras, args.ply_file, args.output_path, args.path_to_project, args.path_to_segmentation)#, args.visualize)


