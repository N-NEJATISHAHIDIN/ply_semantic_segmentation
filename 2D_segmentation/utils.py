import sys, os
import numpy as np
import torch


from PIL import Image
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter

def blur_outside_mask(img, mask_array, blur_radius=15):
    # Convert the NumPy array to a PIL Image
    mask_array = mask_array*255
    mask = Image.fromarray(mask_array.astype(np.uint8))

    # Ensure the mask is in grayscale
    mask = mask.convert("L")  # Convert mask to grayscale

    # Blur the image
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    # import pdb; pdb.set_trace()


    # Create an inverted mask where the original area is 0 and blurred area is 255
    inverted_mask = Image.eval(mask, lambda x: 255 - x)

    # Composite the original image with the blurred image using the inverted mask
    final_img = Image.composite(blurred_img, img, inverted_mask)

    return final_img

def read_images_from_path(path):
    """
    Read all images from the specified path, sorted based on filenames.
    
    Args:
        path (str): The path where the images are located.
    
    Returns:
        List of tuples: (filename, PIL.Image.Image object).
    """
    # Initialize an empty list to store image filenames
    image_files = []
    # Iterate through all files in the directory
    for filename in os.listdir(path):
        # Check if the file is an image (assuming common image file extensions)
        if filename.lower().endswith(('.png', '.jpg.tif', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Append the image filename to the list
            image_files.append(os.path.join(path,filename))
    return image_files

def get_bounding_box(mask,i):
    # Find indices where the mask is True
    
    rows, cols = np.nonzero(mask)
    
    if len(rows) == 0 or len(cols) == 0:
        # No True values in the mask
        return None

    # Determine bounding box coordinates
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    return torch.tensor([ 0, 100*(i+1), 0, 0])


CLASSES = ['airplane', 'banner', 'parking lot', 'construction sight' ,'baseball bat', 'bench', 'bicycle', 'bird', 
               'boat', 'bridge', 'building', 'bus', 'car', 'dirt', 'grass', 'gravel', 
               'house', 'motorcycle', 'mountain','pavement', 'person', 
               'playingfield', 'potted plant', 'railroad', 'road', 'rock', 'sand', 'sea',  
               'sky-other-merged', 'stop sign', 'tent', 'traffic light', 
               'tree', 'truck', 'water', 'grass']


kitti_classes = ['unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 
                 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 
                 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
