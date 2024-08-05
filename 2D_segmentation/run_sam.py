import os
import cv2
import torch
import pickle
import gzip
import argparse
import numpy as np


from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import supervision as sv
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from utils import read_images_from_path



def run_sam(directory, output):
    device = "cuda"

    image_files = read_images_from_path(directory)
    sam = sam_model_registry["vit_h"](checkpoint="/home/negar/secondssd/opendronemap/ply_semantic_segmentation/2D_segmentation/sam_vit_h_4b8939.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)

    for image_file in tqdm(sorted(image_files)):

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resized_image = cv2.resize(image,(int(image.shape[1]), int(image.shape[0])) )
        # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            masks = mask_generator.generate(image)
        pickle.dump(masks, gzip.open(os.path.join(output, os.path.basename(image_file)[:-4])+".pkl", 'wb'))

        detections = sv.Detections.from_transformers(transformers_results={"scores": torch.tensor(np.stack([mask['stability_score'] for mask in masks])).to(device), "boxes" : torch.tensor(np.stack([mask['bbox'] for mask in masks])).to(device), "labels" : torch.arange(len(masks)).to(device)})
        masks_tensor = np.stack([mask['segmentation'] for mask in masks])
        detections.mask = masks_tensor

        mask_annotator = sv.MaskAnnotator()
        # box_annotator = sv.BoxAnnotator()

        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        # annotated_frame = box_annotator.annotate(scene=annotated_image.copy(), detections=detections)#, labels=labels)
        # sv.plot_image(annotated_image, (20, 20))
        cv2.imwrite(os.path.join(output, os.path.basename(image_file)), annotated_image)


  



def main():
    parser = argparse.ArgumentParser(description='Read all images from a specified path.')
    parser.add_argument('path', type=str, help='Path to the directory containing images')
    parser.add_argument('outputpath', type=str, help='Path to the directory containing images')
    args = parser.parse_args()
    
    path = args.path
    if not os.path.exists(path):
        print("Error: The specified path does not exist.")
        return
    
    output_path = args.outputpath
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # import pdb; pdb.set_trace()
    run_sam(path, output_path)


if __name__ == "__main__":
    main()  
