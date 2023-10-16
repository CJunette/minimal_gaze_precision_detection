import os

import torch.cuda

import classification_with_cnn
import clip_with_fasterrcnn
import clip_with_yolo
import exp
import regression_with_cnn
import temporary_functions

if __name__ == "__main__":
    # exp.exp_main()  # 用于实验
    # clip_with_yolo.clip_human_in_batch()  # 使用yolo识别人体，并将图片裁剪为300*300。

    # 用于手动裁切部分有问题的图片。
    # for i in range(34, 44):
    #     temporary_functions.crop_image_manually(0, "horizontal_first", 19, i, 232, 27, 300)

    # 查看图片。
    # temporary_functions.view_pictures([i for i in range(0, 30, 3)], [i for i in range(0, 60, 3)])

    # classification_with_cnn.train_model()
    # regression_with_cnn.train_model()
    regression_with_cnn.validate_model_of_different_districts()


