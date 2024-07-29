# import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt

# # Load point cloud data
# # ply_file_path = '/home/negar/secondssd/opendronemap/datasets/project/odm_filterpoints/point_cloud.ply'
# ply_file_path = '/home/negar/secondssd/opendronemap/datasets/project_neigborhood/odm_filterpoints/point_cloud.ply'
# point_cloud = o3d.io.read_point_cloud(ply_file_path) 

# # Create an Open3D point cloud object
# pcd = o3d.geometry.PointCloud()
# pcd.points = point_cloud.points

# # Perform voxel downsampling (as a form of supervoxel segmentation)
# voxel_size = 0.001
# pcd_down = pcd.voxel_down_sample(voxel_size)

# # Extract voxel clusters
# voxel_labels = np.asarray(pcd_down.cluster_dbscan(eps=voxel_size, min_points=100))

# # Assign colors to clusters
# colors = plt.get_cmap("tab20")(voxel_labels / (voxel_labels.max() if voxel_labels.max() > 0 else 1))
# pcd_down.colors = o3d.utility.Vector3dVector(colors[:, :3])

# # Visualize the segmented point cloud
# o3d.visualization.draw_geometries([pcd_down])


import pclpy
from pclpy import pcl

def generate_supervoxels(point_cloud_file, voxel_resolution=0.008, seed_resolution=0.1, color_importance=0.2,
                         spatial_importance=0.4, normal_importance=1.0):
    # Load the point cloud from file
    cloud = pcl.PointCloud.PointXYZRGBA()
    pcl.io.loadPCDFile(point_cloud_file, cloud)

    # Set up the supervoxel clustering
    supervoxel = pcl.pcl.SupervoxelClustering.PointXYZRGBA(voxel_resolution, seed_resolution)
    supervoxel.setInputCloud(cloud)
    supervoxel.setColorImportance(color_importance)
    supervoxel.setSpatialImportance(spatial_importance)
    supervoxel.setNormalImportance(normal_importance)

    # Extract supervoxels
    supervoxel_clusters = pcl.vectors.MapUint32ToSupervoxel()
    supervoxel.extract(supervoxel_clusters)

    return supervoxel_clusters

# Usage example
point_cloud_file =  '/home/negar/secondssd/opendronemap/datasets/project_neigborhood/odm_filterpoints/point_cloud.ply'
supervoxel_clusters = generate_supervoxels(point_cloud_file)

# Print out the number of supervoxels
print(f"Number of supervoxels: {len(supervoxel_clusters)}")