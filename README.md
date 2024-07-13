# point-cloud semantic segmentation
This script performs semantic segmentation on the output point cloud from OpenDroneMap. 

### Steps:
1. Run a segmentation module on all images (one frame).
2. Color same class names with the same color.
3. For every point in the 3D point-cloud, project it into all images and make a list of the point colors. Remove the occluded points.
4. At the end, choose the color for each point which is the most repeated in the point cloud.

### Issues:
1. Incorrect segmentations.
3. Mismatched classes for the same objects in different frames.

## Datasets
This project uses two main, first the semanticly segmented image, then the output directory of opendrownmap (ODM), please download the following datas.

1. **2D Semantic Segmentations**
   - Path: `https://drive.google.com/file/d/1xQtwe3Y7CgeFi2kQhLUNp_GfbGL8OXfu/view?usp=sharing`
   - Semantic segmentations of the RGB images of the neighborhood data.

2. **ODM output**
   - Path: `/path/to/dataset2`
   - the ODM output for neighborhood data.

## Run the semantic segmentation on the pointcloud 

If you want to generate the semanticly segmented pointcloud:
```sh
python segment_pointcloud_clean.py \
path_to_final_ply \ #(path_to_ODM_output/odm_filterpoints/point_cloud.ply)
path_to_camera_information\ #(path_to_ODM_output/opensfm/camera_models.json)
path_to_2D_segmentations\
path_to_ODM_output\
path_to_output\ # give name of a folder which you want the output saved to 
```

<!-- If you also want to visualize the images per view:
```sh
python segment_pointcloud_clean.py \
path_to_final_ply \ #(path_to_ODM_output/odm_filterpoints/point_cloud.ply)
path_to_camera_information\ #(path_to_ODM_output/opensfm/camera_models.json)
path_to_2D_segmentations\
path_to_ODM_output\
path_to_output\ # give name of a folder which you want the output saved to 
visualize -->
```
My command is: 

```sh
python segment_pointcloud_clean.py './project_neigborhood/odm_filterpoints/point_cloud.ply' './project_neigborhood/opensfm/reconstruction.json' './segmentations' './project_neigborhood' ./output_ply 
```
