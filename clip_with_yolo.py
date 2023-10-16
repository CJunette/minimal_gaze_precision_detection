import os
from multiprocessing import Pool
from torchvision.transforms import ToPILImage
import numpy as np
import torch
from pathlib import Path
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

import configs
import sys

# original_sys_path = sys.path.copy()
# sys.path.append('./models/yolov5')
# from models.yolo import Model
# sys.path = original_sys_path


def crop_image_with_torch(img_tensor, center_x, center_y, half_width):
    """
    Crop an image tensor using PyTorch.
    """
    # Calculate the cropping coordinates
    _, c, h, w = img_tensor.shape
    top = int(center_y - half_width)
    left = int(center_x - half_width)
    bottom = int(center_y + half_width)
    right = int(center_x + half_width)

    # Ensure coordinates are within image boundaries
    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)

    return img_tensor[:, :, top:bottom, left:right]


def crop_image_single_pool(img_tensor, pred, image_path, image, output_path):
    print(image_path)
    # Filter predictions for the 'person' class with confidence > 0.8
    person_preds = [x for x in pred if x[4] > 0.8 and int(x[5]) == 0]  # '0' is the class index for 'person' in COCO dataset
    if not person_preds:
        print(f"No person detected in image {image_path}")
        return

    box = person_preds[0][:4]  # Get the highest confidence prediction
    head_shoulder_height = (box[3] - box[1]) * 0.5
    center_x = (box[0] + box[2]) / 2
    center_y = box[1] - head_shoulder_height

    half_width = 150
    cropped_image = image.crop((center_x - half_width, center_y - half_width, center_x + half_width, center_y + half_width))

    # cropped_tensor = crop_image_with_torch(img_tensor, center_x, center_y, half_width)
    # a = cropped_tensor.cpu().numpy()
    # cropped_image = Image.fromarray((cropped_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

    # plt.imshow(cropped_image)
    # plt.show()

    # output_path = output_path
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # cropped_image.save(f"{output_path}capture{image_path.split('capture')[1]}")

    return cropped_image


def process_images(model, device, image_path_list, output_path_list):
    images = [Image.open(p) for p in image_path_list]
    images_tensor = [torchvision.transforms.ToTensor()(img).unsqueeze(0) for img in images]  # Convert to tensor
    images_tensor = [img_tensor.to(device) for img_tensor in images_tensor]  # Move tensors to GPU if available
    images_batch = torch.cat(images_tensor, dim=0)

    results = model(images_batch)

    object_masks = results[:, :, 4] > 0.8
    class_masks = torch.argmax(results[:, :, 5:], dim=2) == 0
    final_masks = object_masks & class_masks

    person_preds = [None for _ in range(results.shape[0])]
    boxes = [None for _ in range(results.shape[0])]

    for i, pred in enumerate(results):
        person_pred = pred[final_masks[i]]
        if person_pred.shape[0] == 0:
            print(f"No person detected in image {image_path_list[i]}")
            continue
        else:
            person_preds[i] = person_pred[0]
            boxes[i] = person_pred[0][:4].cpu().numpy()
    boxes = np.array(boxes)

    head_shoulder_heights = (boxes[:, 3] - boxes[:, 1]) * 0.5
    center_xs = ((boxes[:, 0] + boxes[:, 2]) / 2).astype(int)
    center_ys = (boxes[:, 1] - head_shoulder_heights).astype(int)

    center_xs_tensor = torch.from_numpy(center_xs).to(device)
    center_ys_tensor = torch.from_numpy(center_ys).to(device)
    half_width = 150
    half_width_tensor = torch.full_like(center_xs_tensor, half_width)

    x1s = center_xs_tensor - half_width_tensor
    x2s = center_xs_tensor + half_width_tensor
    y1s = center_ys_tensor - half_width_tensor
    y2s = center_ys_tensor + half_width_tensor

    to_pil = ToPILImage()

    cropped_images = [None for _ in range(results.shape[0])]
    for i, img in enumerate(images_tensor):
        if y1s[i] < 0 or x1s[i] < 0 or y2s[i] > img.shape[2] or x2s[i] > img.shape[3]:
            print(f"Image {image_path_list[i]} is out of bounds.")
            cropped_tensor = torch.zeros(3, half_width * 2, half_width * 2).to(img.device)
            y1, y2 = max(y1s[i], 0), min(y2s[i], img.shape[2])
            x1, x2 = max(x1s[i], 0), min(x2s[i], img.shape[3])
            start_y = max(0, -y1s[i])
            start_x = max(0, -x1s[i])
            end_y = start_y + (y2 - y1)
            end_x = start_x + (x2 - x1)
            cropped_tensor[:, start_y:end_y, start_x:end_x] = img[0, :, y1:y2, x1:x2]
            cropped = cropped_tensor
        else:
            cropped = img[0, :, y1s[i]:y2s[i], x1s[i]:x2s[i]]

        cropped_image = to_pil(cropped.cpu())
        cropped_images[i] = cropped_image

    for i in range(len(cropped_images)):
        output_path = output_path_list[i]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cropped_images[i].save(f"{output_path}capture{image_path_list[i].split('capture')[1]}")


def read_model_and_device():
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
    os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:10809'

    # model = Model(cfg='models/yolov5/models/yolov5s.yaml')
    # weights = torch.load("models/yolov5/yolov5_weight/yolov5s.pt")
    # model.load_state_dict(weights)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = torch.load("models/yolov5/yolov5_weight/yolov5s.pt", map_location=torch.device(device))
    model = torch.hub.load("models/yolov5", "custom", "models/yolov5/yolov5_weight/yolov5m.pt", source="local")

    model.to(device)
    model.eval()

    return model, device

def clip_human_in_batch():
    model, device = read_model_and_device()

    image_name_list_1 = []
    image_path_list_1 = []
    file_path_1 = f"output/subject_{configs.subject_num}/{configs.mode}/"
    file_names_1 = os.listdir(file_path_1)
    for file_name_1 in file_names_1:
        if file_name_1.startswith("row_"):
            image_name_list_2 = []
            image_path_list_2 = []
            file_path_2 = f"{file_path_1}{file_name_1}/"
            file_names_2 = os.listdir(file_path_2)
            for file_name_2 in file_names_2:
                file_name_3 = f"{file_path_2}{file_name_2}"
                image_name_list_2.append(file_name_3)
                file_path_3 = f"{file_path_2}".replace(f"{configs.mode}", f"clipped_{configs.mode}")
                image_path_list_2.append(file_path_3)

            image_name_list_1.append(np.array(image_name_list_2))
            image_path_list_1.append(np.array(image_path_list_2))

    image_name_list_1 = np.array(image_name_list_1)
    image_path_list_1 = np.array(image_path_list_1)

    image_name_list_1d = image_name_list_1.reshape(-1)
    image_path_list_1d = image_path_list_1.reshape(-1)

    repeat_times = 180
    single_time_amount = len(image_name_list_1d) // repeat_times
    for i in range(68, repeat_times):
        print(i)
        process_images(model, device, image_name_list_1d[i * single_time_amount: (i + 1) * single_time_amount], image_path_list_1d[i * single_time_amount: (i + 1) * single_time_amount])

