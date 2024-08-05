import os
import clip
import torch
from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image
import pickle

import cv2
import argparse
from tqdm import tqdm

from utils import read_images_from_path, blur_outside_mask, CLASSES, kitti_classes
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_clip(path, sam_path, output_path):

    image_files = read_images_from_path(path)
    model, preprocess = clip.load('ViT-B/32', device)

    # import pdb; pdb.set_trace()

    # Download the dataset
    # cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in kitti_classes]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    for image_file in tqdm(sorted(image_files)):
        text_features_new = text_features
        image = Image.open(image_file)
        image_sam = Image.open(os.path.join(sam_path, os.path.basename(image_file)))
        # image_sam.save(os.path.join(output_path, os.path.basename(image_file)))

        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in kitti_classes]).to(device)
        data = pickle.load(open((os.path.join(sam_path, os.path.basename(image_file))[:-4] + ".pkl"), 'rb'))
        
        for idx, mask in enumerate(data):
            # import pdb; pdb.set_trace()

            new_image = blur_outside_mask(image, mask["segmentation"] )
            # new_image.save(os.path.join(output_path, os.path.basename(image_file))[:-4] + "{}.JPG".format(idx))

            bbox = mask["bbox"]

            cropped_img = new_image.crop((max([bbox[0]- (int(bbox[2]/4)),0]), 
                                          max([bbox[1]- (int(bbox[3]/4)),0]),
                                          min([bbox[0] + bbox[2] + (int(bbox[2]/4)), mask["segmentation"].shape[1]]),
                                          min([bbox[1] + bbox[3] + (int(bbox[3]/4)), mask["segmentation"].shape[0]])))
            image_input = preprocess(cropped_img).unsqueeze(0).to(device)


            # Calculate features
            with torch.no_grad():
                image_features = model.encode_image(image_input)

            # Pick the top 5 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features_new /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features_new.T).softmax(dim=-1)
            values, indices = similarity[0].topk(10)

            # Print the result
            preds = ""
            # print("\nTop predictions:\n")
            for value, index in zip(values, indices):
                preds += f"_{kitti_classes[index]}:{100 * value.item():.2f}%"
            # import pdb; pdb.set_trace()

            # cropped_img_sam = image_sam.crop((bbox[0], bbox[1],
            #                             bbox[0] + bbox[2],
            #                             bbox[1] + bbox[3]))
            out_path = os.path.join(output_path, os.path.basename(image_file)[:-4])
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            cropped_img.save(os.path.join(out_path,'{}{}.jpeg'.format(idx, preds)))
            # import pdb; pdb.set_trace()




def main():
    parser = argparse.ArgumentParser(description='Read all images from a specified path.')
    parser.add_argument('path', type=str, help='Path to the directory containing images')
    parser.add_argument('sam_path', type=str, help='Path to the directory containing sam outputs')
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
    run_clip(path, args.sam_path, output_path)

if __name__ == "__main__":
    main()