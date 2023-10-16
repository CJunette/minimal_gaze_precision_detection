import os

import matplotlib.pyplot as plt
from PIL import Image

import torchvision
from torchvision.transforms import ToPILImage

import configs


def crop_image_manually(subject_index, mode, row_index, col_index, x_min, y_min, crop_width):
    input_path = f"output/subject_{subject_index}/{mode}/row_{row_index}-col_{col_index}/"
    output_path = f"output/subject_{subject_index}/clipped_{mode}/row_{row_index}-col_{col_index}/"
    to_pil = ToPILImage()

    input_files = os.listdir(input_path)
    for input_file in input_files:
        if input_file.endswith(".jpg"):
            input_image = Image.open(input_path + input_file)
            image_tensor = torchvision.transforms.ToTensor()(input_image).unsqueeze(0)
            cropped = image_tensor[0, :, y_min:y_min+crop_width, x_min:x_min+crop_width]
            cropped_image = to_pil(cropped.cpu())

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            cropped_image.save(f"{output_path}{input_file}")


def view_pictures(row_list, col_list):
    fig, axs = plt.subplots(len(row_list), len(col_list))
    # 设置plt分辨率、图像间的间距及图像到周围的距离
    fig.set_size_inches(64, 36)
    fig.subplots_adjust(wspace=0.05, hspace=0.025, left=0.05, right=0.95, top=0.95, bottom=0.05)

    for row_index in range(len(row_list)):
        for col_index in range(len(col_list)):
            row = row_list[row_index]
            col = col_list[col_index]
            print(row_index, col_index)
            input_path = f"output/subject_{configs.subject_num}/clipped_{configs.mode}/row_{row}-col_{col}/"
            input_files = os.listdir(input_path)
            for i, input_file in enumerate(input_files):
                if i > 0:
                    break
                if input_file.endswith(".jpg"):
                    input_image = Image.open(input_path + input_file)
                    axs[row_index, col_index].imshow(input_image)
    # plt.show()
    # save plt with 3200*1800 resolution
    plt.savefig(f"preview_row_{len(row_list)}_col_{len(col_list)}.jpeg", dpi=300)
