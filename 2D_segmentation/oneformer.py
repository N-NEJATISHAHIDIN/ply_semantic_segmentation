from transformers import AutoProcessor
from transformers import AutoModelForUniversalSegmentation

from PIL import Image
import torch
import supervision as sv
import numpy as np
import cv2
import os, sys
from utils import read_images_from_path, get_bounding_box
from tqdm import tqdm



# the Auto API loads a OneFormerProcessor for us, based on the checkpoint
processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")


images = read_images_from_path('/home/negar/secondssd/opendronemap/datasets/project/images')
output_path = "./results/visualize_dino_sam_oneformer_manassas_all_oneformer_classes"
dino_img_path= "/home/negar/secondssd/opendronemap/ply_semantic_segmentation/2D_segmentation/results/dino_sam_all_oneformer_classes_visual"
if not os.path.exists(output_path):
    os.makedirs(output_path)
for image_path in tqdm(sorted(images)):
    print(image_path)
    model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")
    image = Image.open(image_path)
    print(image.size)
    scene = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    # prepare image for the model
    panoptic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")


    # forward pass
    with torch.no_grad():
        outputs = model(**panoptic_inputs)
    panoptic_segmentation = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    classes = model.config.id2label

    scores = torch.asarray([i['score'] for i in panoptic_segmentation['segments_info']])
    label_id = torch.tensor([i['label_id'] for i in panoptic_segmentation['segments_info']]).to('cuda')
    masks = torch.stack([panoptic_segmentation['segmentation']==i['id'] for i in panoptic_segmentation['segments_info']])
    boxes = torch.stack([get_bounding_box(np.asarray(masks[i]),i) for i in range(len(panoptic_segmentation['segments_info']))])
    print("###################\n", len(panoptic_segmentation['segments_info']), np.unique(panoptic_segmentation['segmentation']), 
          "###################\n", panoptic_segmentation['segments_info'])


    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    detections = sv.Detections.from_transformers(transformers_results={"scores": scores, "boxes" : boxes, "labels" : label_id})
    masks_np = masks.cpu().numpy()
    detections.mask = masks_np


    labels = []
    for _, _, confidence, class_id, _  in detections:
        labels.append(f"{classes[class_id]} {confidence:0.2f}")

    annotated_image = mask_annotator.annotate(scene=scene.copy(), detections=detections)
    annotated_frame = (box_annotator.annotate(scene=annotated_image.copy(), detections=detections, labels=labels))#,  np.hstack(np.ones((2250, 250,3),dtype = int)*255))
    
    # import pdb; pdb.set_trace()
    # sv.plot_image(annotated_frame, (20, 20))
    try:
        # import pdb; pdb.set_trace()
        image_dino = Image.open(os.path.join(dino_img_path,os.path.basename(image_path)))
        scene_dino = cv2.cvtColor(np.asarray(image_dino), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, os.path.basename(image_path)), np.vstack((scene,annotated_frame,scene_dino)))
    except:
        print("image_not exist for dino-sam")
        continue