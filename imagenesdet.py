import argparse
import os
import cv2
import torch
from utils.datasets import *
from utils.utils import *

def detect_images_in_folder(model, folder_path, class_names, conf_thres):
    model.eval()

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            imgTensor = transforms.ToTensor()(img)
            imgTensor, _ = pad_to_square(imgTensor, 0)
            imgTensor = resize(imgTensor, 416)
            imgTensor = imgTensor.unsqueeze(0)

            with torch.no_grad():
                detections = model(imgTensor)
                detections = non_max_suppression(detections, conf_thres, 0.4)

            for detection in detections:
                if detection is not None:
                    detection = rescale_boxes(detection, 416, img.shape[:2])
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                        color = (0, 255, 0)
                        label = f"{class_names[int(cls_pred)]}: {conf:.2f}"
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Image", img)
            cv2.waitKey(0)

if __name__ == "__main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, required=True, help="Path to model definition file")
    parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    parser.add_argument("--class_path", type=str, required=True, help="Path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.85, help="Object confidence threshold")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to image folder")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(args.model_def).to(device)
    model.load_state_dict(torch.load(args.checkpoint_model))
    model.eval()

    with open(args.class_path, "r") as f:
        class_names = f.read().strip().split("\n")

    detect_images_in_folder(model, args.image_folder, class_names, args.conf_thres)

