import os
import requests
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import supervision as sv
import cv2
from ultralytics import YOLO, SAM

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class Detections:
    def __init__(self, boxes, labels, scores):
        self.xyxy = boxes
        self.labels = labels
        self.scores = scores
        self.class_id = 1  # Assuming labels can be used as class_ids

    def __len__(self):
        return len(self.xyxy)

    def __getitem__(self, idx):
        return self.xyxy[idx], self.labels[idx], self.scores[idx]

def run_dino_sam(directory, output):
    device = "cuda"
    model_id = "IDEA-Research/grounding-dino-tiny"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    sam_predictor = SAM('mobile_sam.pt') # UltraLytics SAM

    image_files = read_images_from_path(directory)
    text = "a house. a tree. a car. a road. "
    classes = text.split(". ")
    class_to_id = {}
    for id, class_name in enumerate(classes):
        class_to_id[class_name] = id
    for image_file in tqdm(sorted(image_files)):

        image = Image.open(image_file)

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.22,
            text_threshold=0.2,
            target_sizes=[image.size[::-1]]
        )[0]

        image = np.asarray(image)
        scene = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        detections = sv.Detections.from_transformers(transformers_results={"scores": results["scores"], "boxes" : results["boxes"], "labels" : torch.tensor([class_to_id[class_name] for class_name in results["labels"]]).to("cuda") })

        
        # Annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _  in detections]
        # import pdb; pdb.set_trace()

        annotated_frame = box_annotator.annotate(scene=scene.copy(), detections=detections, labels=labels)
        sv.plot_image(annotated_frame, (16, 16))
        # import pdb; pdb.set_trace()

        sam_out = sam_predictor.predict(image_file, bboxes=results["boxes"], verbose=False)
        masks_tensor = sam_out[0].masks.data
        masks_np = masks_tensor.cpu().numpy()
        detections.mask = masks_np


        mask_annotator = sv.MaskAnnotator()
        annotated_image = mask_annotator.annotate(scene=annotated_frame.copy(), detections=detections)
        # annotated_image = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        sv.plot_image(annotated_image, (16, 16))



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
    
    run_dino_sam(path, output_path)
    # images = read_images_from_path(path)
    # print(f"Found {len(images)} images in {path}.")

if __name__ == "__main__":
    main()  
