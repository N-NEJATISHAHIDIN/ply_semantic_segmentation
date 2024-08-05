import os
import requests
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import supervision as sv
import cv2
from ultralytics import YOLO, SAM

import pickle
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils import read_images_from_path
from transformers import AutoProcessor
from transformers import AutoModelForUniversalSegmentation

# processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
# modelee = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")
# classes = modelee.config.label2id


def generate_k_distinct_colors(k):
    # Ensure k is at most 12 to avoid non-distinct colors
    k = min(k, 12)

    # Generate k evenly spaced angles in hue space
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)

    # Convert these angles to RGB colors
    colors = np.array([
        [np.sin(angle), np.sin(angle + 2 * np.pi / 3), np.sin(angle + 4 * np.pi / 3)]
        for angle in angles
    ])

    # Normalize colors to be in [0, 1] range
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    
    return colors

# Example usage


def run_dino_sam(directory, text, output):
    device = "cuda"
    model_id = "IDEA-Research/grounding-dino-tiny"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    sam_predictor = SAM('sam_l.pt') # UltraLytics SAM

    image_files = read_images_from_path(directory)
    # text = "a house. a car. a road. a sidewalk. "
    classes = text.split(". ")[:-1]
    # classes.append("not single class")
    class_to_id = {}
    # k = len(classes)
    # colors = generate_k_distinct_colors(k)

    for id, class_name in enumerate(classes):
        class_to_id[class_name] = id
    for image_file in tqdm(sorted(image_files)):

        image = Image.open(image_file)

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        # import pdb; pdb.set_trace()

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.30,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )[0]

        image = np.asarray(image)
        # import pdb; pdb.set_trace()
        scene = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # try:
        label_id = []
        for class_name in results["labels"]:
            try:
                label_id.append(class_to_id[class_name])
            except:
                try:
                    id = np.where([(classe in class_name) for classe in classes])[0][0]
                    label_id.append(id)

                except:

                    print(class_name)

                    # import pdb; pdb.set_trace()
                # class_to_id[classes] = len(class_to_id)
                # label_id.append(class_to_id[class_name])
                    label_id.append(0)
                # classes.append(class_name)
        # print("label_id: ", label_id)
        # print("classes: ", classes)

        # except:
        #     print(results["labels"])
        #     print("Warning: passing image {}".format(os.path.basename(image_file)))
        #     import pdb; pdb.set_trace()
        #     continue

        detections = sv.Detections.from_transformers(transformers_results={"scores": results["scores"], "boxes" : results["boxes"], "labels" : torch.tensor(label_id).to("cuda") })

        # Annotate image with detections
        box_annotator = sv.BoxAnnotator()
        # labels = [
        #     f"{classes[class_id]} {confidence:0.2f}" 
        #     for _, _, confidence, class_id, _  in detections]
        labels = []
        for _, _, confidence, class_id, _  in detections:

            labels.append(f"{classes[class_id]} {confidence:0.2f}")

        
        # import pdb; pdb.set_trace()

        # annotated_frame = box_annotator.annotate(scene=scene.copy(), detections=detections, labels=labels)
        # sv.plot_image(annotated_frame, (16, 16))
        try:
            sam_out = sam_predictor.predict(image_file, bboxes=results["boxes"], verbose=False)
        except:
            continue
        masks_tensor = sam_out[0].masks.data
        masks_np = masks_tensor.cpu().numpy()
        detections.mask = masks_np
        # with open('{}/{}.pkl'.format(output, os.path.basename(image_file)[:-4]), 'wb') as file:
        #     pickle.dump(detections, file)
        # import pdb; pdb.set_trace()
        black_image = np.ones((image.shape),dtype = 'uint8')*200
        black_image = cv2.cvtColor(black_image, cv2.COLOR_RGB2BGR)
        mask_annotator = sv.MaskAnnotator()
        annotated_image = mask_annotator.annotate(scene=scene.copy(), detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_image.copy(), detections=detections, labels=labels)

        # annotated_frame = box_annotator.annotate(scene=annotated_image.copy(), detections=detections, labels=labels)

        # annotated_image = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        # sv.plot_image(annotated_frame, (20, 20))
        cv2.imwrite(os.path.join(output, os.path.basename(image_file)), annotated_frame)


    #     # generate mask for the ply segmentation generation
    #     mask_annotator_no_image = sv.MaskAnnotator()
    #     black_image = np.zeros((image.shape),dtype = 'uint8')
    #     black_image = cv2.cvtColor(black_image, cv2.COLOR_RGB2BGR)
    #     # import pdb; pdb.set_trace()
    #     annotated_image = mask_annotator_no_image.annotate(scene=black_image.copy(), detections=detections)

    #     cv2.imwrite(os.path.join(output, os.path.basename(image_file)), annotated_image)
    # # with open('{}/{}.pkl'.format(output, "classes_to_ids"), 'wb') as file:
    # #     pickle.dump(class_to_id, file)






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
    # text = "a sedan car. an suv car. a tree. a light. a fan. a box."
    text = "a car. all cars. all roads. all buildings. a tree. a light. all sidewalks. all grass. all construction materials. "
    # text = "a house. a car. a road. a sidewalk. all grass. a tree. "
    # import pdb; pdb.set_trace()
    classes = ['airplane', 'banner', 'parking', 'construction' ,'baseball bat', 'baseball glove', 'bench', 'bicycle', 'bird', 
               'boat', 'bridge', 'building', 'bus', 'car', 'dirt-merged', 'fire hydrant', 'frisbee', 'grass-merged', 'gravel', 
               'horse', 'house', 'kite', 'motorcycle', 'mountain-merged', 'parking meter', 'pavement', 'person', 'platform', 
               'playingfield', 'potted plant', 'railroad', 'road', 'rock-merged', 'roof', 'sand', 'sea', 'sheep', 'skateboard', 
               'skis', 'sky-other-merged', 'snow', 'snowboard', 'sports ball', 'stop sign', 'surfboard', 'tent', 'traffic light', 
               'train', 'tree-merged', 'truck', 'water-other', 'tree', 'grass', 'jungle']

    run_dino_sam(path, ". ".join(classes)+". ", output_path)
    # images = read_images_from_path(path)
    # print(f"Found {len(images)} images in {path}.")

if __name__ == "__main__":
    main()  
