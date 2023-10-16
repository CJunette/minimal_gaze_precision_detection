import torch
import torchvision.transforms as T
from matplotlib import pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import numpy as np
import configs
import os


def clip_human_and_save(src_path, location_index_str, capture_index_str, model, device):
    # 2. 加载图片
    image = Image.open(f"{src_path}").convert("RGB")
    # 3. 定义图像转换
    transform = T.Compose([T.ToTensor()])

    # 4. 应用转换并添加批处理维度
    input_image = transform(image).unsqueeze(0)
    input_image = input_image.to(device)

    # 6. 执行推理
    with torch.no_grad():
        prediction = model(input_image)

    # 7. 提取置信度分数>0.8的“人”类的边界框 (标签id=1)
    person_boxes = prediction[0]["boxes"][(prediction[0]["labels"] == 1) & (prediction[0]["scores"] > 0.8)].cpu().numpy()

    # 8. 考虑图像中检测到的第一个人（如果有）
    if len(person_boxes) > 0:
        box = person_boxes[0]
        head_shoulder_height = (box[3] - box[1]) * 0.5

        center_x = (box[0] + box[2]) / 2
        center_y = box[1] + head_shoulder_height / 2

        half_width = 200
        cropped_image = image.crop((center_x - half_width, center_y - half_width,
                                    center_x + half_width, center_y + half_width))

        # plt.imshow(cropped_image)
        # plt.show()
        save_path = f"output/subject_{configs.subject_num}/clipped_{configs.mode}/{location_index_str}/{capture_index_str}"
        cropped_image.save(save_path)

    else:
        print("No person detected in the image.")


# def clip_human_in_batch():
#     file_path = f"output/subject_{configs.subject_num}/{configs.mode}/"
#
#     # 1. 加载预训练模型并设置为评估模式
#     model = fasterrcnn_resnet50_fpn(pretrained=True)
#     model.eval()
#
#     # 5. 如果CUDA可用，将模型和输入数据移动到GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#     file_names_1 = os.listdir(file_path)
#     for file_name_1 in file_names_1:
#         if file_name_1.startswith("row_"):
#             file_path_1 = file_path + file_name_1 + "/"
#             file_names_2 = os.listdir(file_path_1)
#             for file_name_2 in file_names_2:
#                 file_name_3 = file_path_1 + file_name_2
#                 if file_name_3.endswith(".jpg"):
#                     clip_human_and_save(file_name_3, file_name_1, file_name_2, model, device)


def load_clip_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def process_images(image_paths, model, device, output_path_list):
    transform = T.Compose([T.ToTensor()])

    # Load images
    images = [Image.open(path).convert("RGB") for path in image_paths]
    input_images = torch.stack([transform(img) for img in images]).to(device)

    # Inference
    with torch.no_grad():
        predictions = model(input_images)

    for i, prediction in enumerate(predictions):
        person_boxes = prediction["boxes"][(prediction["labels"] == 1) & (prediction["scores"] > 0.8)].cpu().numpy()
        if len(person_boxes) == 0:
            print(f"No person detected in the image {image_paths[i]}.")
            continue

        box = person_boxes[0]
        head_shoulder_height = (box[3] - box[1]) * 0.5

        center_x = (box[0] + box[2]) / 2
        center_y = box[1] + head_shoulder_height / 2

        half_width = 200
        cropped_image = images[i].crop((center_x - half_width, center_y - half_width, center_x + half_width, center_y + half_width))
        # Save the processed image
        output_path = output_path_list[i]
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        cropped_image.save(f"{output_path}capture{image_paths[i].split('capture')[1]}")


def clip_human_in_batch():
    model, device = load_clip_model()

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

    repeat_times = 1800
    single_time_amount = len(image_name_list_1d) // repeat_times
    for i in range(repeat_times):
        process_images(image_name_list_1d[i * single_time_amount : (i + 1) * single_time_amount], model, device, image_path_list_1d[i * single_time_amount : (i + 1) * single_time_amount])

