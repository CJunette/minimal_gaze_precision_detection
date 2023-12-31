import random
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import torch.nn as nn
import configs
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


# def compute_average_picture(pics):
#     avg_pic = np.mean(pics)
#     return avg_pic


def custom_sort(class_name):
    row_val, col_val = map(int, class_name.replace("row_", "").replace("col_", "").split('-'))
    return row_val, col_val


class CustomDataset(Dataset):
    def __init__(self, root_dir, file_path_list, transform=None, train=True, horizontal_step=1, vertical_step=1):
        self.transform = transform
        self.train = train

        self.image_paths = []
        self.labels = []
        for class_name in file_path_list:
            row_val, col_val = map(int, class_name.replace("row_", "").replace("col_", "").split('-'))
            image_names = sorted(os.listdir(os.path.join(root_dir, class_name)))
            if self.train:
                # 如果是训练数据，则选择每个类别的第1张图片
                valid_image_name_list = [image_names[0], image_names[3], image_names[6]]
            else:
                # 如果是验证数据，则选择每个类别的第4和第7张图片
                valid_image_name_list = [image_names[1], image_names[4], image_names[7]]

            for image_name in valid_image_name_list:
                self.image_paths.append(os.path.join(root_dir, class_name, image_name))
                self.labels.append((row_val, col_val))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.float32)


class SimpleCNNModel(nn.Module):
    def __init__(self):
        super(SimpleCNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 32 * 32, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 2),  # Output 2 values representing x and y coordinates
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleCNNLightning(SimpleCNNModel, pl.LightningModule):
    def __init__(self):
        super(SimpleCNNLightning, self).__init__()
        self.validation_outputs = []
        self.training_outputs = []

    def training_step(self, batch, batch_idx):
        images, coordinates = batch
        outputs = self(images)
        loss = nn.MSELoss()(outputs, coordinates)
        mse_loss = loss.item()
        self.training_outputs.append(mse_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, coordinates = batch
        outputs = self(images)
        loss = nn.MSELoss()(outputs, coordinates)
        mse_loss = loss.item()
        self.validation_outputs.append(mse_loss)
        return {'val_loss': loss}

    def on_train_epoch_end(self):
        avg_mse_loss = np.mean(self.training_outputs)
        self.log('train_mse_loss', avg_mse_loss)

    def on_validation_epoch_end(self):
        avg_mse_loss = np.mean(self.validation_outputs)
        self.log('val_mse_loss', avg_mse_loss)

    def on_train_epoch_start(self):
        self.training_outputs.clear()

    def validation_epoch_start(self):
        self.validation_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)


def prepare_data_select_with_step(horizontal_step=1, vertical_step=1):
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    root_dir = f'output/subject_{configs.subject_num}/clipped_{configs.mode}'
    file_path_list = os.listdir(root_dir)
    file_path_list.sort(key=custom_sort)

    # 对classes进行分类，row_x相同的为一组。
    grouped_file_path_list = [file_path_list[i:i + configs.col_num] for i in range(0, len(file_path_list), configs.col_num)]
    for i in range(len(grouped_file_path_list)):
        grouped_file_path_list[i] = grouped_file_path_list[i][::horizontal_step]
    grouped_file_path_list = grouped_file_path_list[::vertical_step]
    file_path_list = np.array(grouped_file_path_list).reshape(-1)

    train_dataset = CustomDataset(root_dir=root_dir, file_path_list=file_path_list, transform=transform, train=True, horizontal_step=horizontal_step, vertical_step=vertical_step)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = CustomDataset(root_dir=root_dir, file_path_list=file_path_list, transform=transform, train=False, horizontal_step=horizontal_step, vertical_step=vertical_step)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


def prepare_data_within_block(horizontal_block, vertical_block):
    horizontal_block_step = ceil(configs.col_num / horizontal_block)
    vertical_block_step = ceil(configs.row_num / vertical_block)

    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    root_dir = f'output/subject_{configs.subject_num}/clipped_{configs.mode}'
    file_path_list = os.listdir(root_dir)
    file_path_list.sort(key=custom_sort)

    # 对classes进行分类，row_x相同的为一组。
    grouped_file_path_list = [file_path_list[i:i + configs.col_num] for i in range(0, len(file_path_list), configs.col_num)]

    train_loader_list = []
    val_loader_list = []
    train_dataset_list = []
    val_dataset_list = []
    # train_tag_list = []
    # val_tag_list = []
    train_group_tag_list = []
    val_group_tag_list = []

    for horizontal_index in range(horizontal_block):
        for vertical_index in range(vertical_block):
            print(f"preparing data: horizontal-{horizontal_index} vertical-{vertical_index}")
            file_path_of_block = []
            # tag_list = []
            for vertical_step_index in range(vertical_block_step):
                for horizontal_step_index in range(horizontal_block_step):
                    file_path_of_block.append(grouped_file_path_list[vertical_index * vertical_block_step + vertical_step_index][horizontal_index * horizontal_block_step + horizontal_step_index])
                    # tag_list.append((vertical_index * vertical_block_step + vertical_step_index, horizontal_index * horizontal_block_step + horizontal_step_index))
            file_path_of_block = np.array(file_path_of_block).reshape(-1)

            train_dataset = CustomDataset(root_dir=root_dir, file_path_list=file_path_of_block, transform=transform, train=True)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            val_dataset = CustomDataset(root_dir=root_dir, file_path_list=file_path_of_block, transform=transform, train=False)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            train_loader_list.append(train_loader)
            val_loader_list.append(val_loader)
            train_dataset_list.append(train_dataset)
            val_dataset_list.append(val_dataset)
            # train_tag_list.append(tag_list)
            # val_tag_list.append(tag_list)
            train_group_tag_list.append((vertical_index, horizontal_index))
            val_group_tag_list.append((vertical_index, horizontal_index))

    return train_loader_list, val_loader_list, train_dataset_list, val_dataset_list, train_group_tag_list, val_group_tag_list


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model():
    set_seed(42)

    train_loader, val_loader = prepare_data_select_with_step(1, 1)
    num_epochs = configs.num_epochs

    # 从头训练。
    model = SimpleCNNLightning()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mse_loss',
        dirpath=f'models/custom_regression_cnn/{configs.custom_model_name}',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )
    trainer = pl.Trainer(max_epochs=num_epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)

    # 接续上次训练。
    # checkpoint_path = f"models/custom_regression_cnn/{configs.custom_model_name}/best-checkpoint.ckpt"
    # model = SimpleCNN()
    # logger = TensorBoardLogger(save_dir="lightning_logs", name="", version="version_3")
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_mse_loss',
    #     dirpath=f'models/custom_cnn/{configs.custom_model_name}',  # 明确指定保存目录
    #     filename='best-checkpoint',
    #     save_top_k=1,
    #     mode='min'
    # )
    # trainer = pl.Trainer(max_epochs=num_epochs, default_root_dir="lightning_logs", logger=logger, callbacks=[checkpoint_callback])
    #
    # trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)


def validate_model_of_different_districts(horizontal_blocks, vertical_blocks):
    set_seed(42)

    train_loader, val_loader = prepare_data_select_with_step()
    checkpoint = torch.load(f"models/custom_regression_cnn/{configs.custom_model_name}/best-checkpoint.ckpt")
    model_state_dict = checkpoint['state_dict']
    model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items()}

    model = SimpleCNNModel()
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()

    prediction_list = []
    precision_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()  # Move images to GPU
            outputs = model(images)
            prediction_list.extend(outputs.cpu().numpy())
            precision_1 = labels.cpu().numpy() - outputs.cpu().numpy()
            precision_2 = np.sqrt(np.sum(precision_1 ** 2, axis=1))
            precision_list.extend(precision_2)

    horizontal_block_step = ceil(configs.col_num / horizontal_blocks)
    vertical_block_step = ceil(configs.row_num / vertical_blocks)
    indices_of_block_list_1 = []
    for vertical_index in range(vertical_blocks):
        for horizontal_index in range(horizontal_blocks):
            print(f"preparing data: vertical-{vertical_index} horizontal-{horizontal_index}")
            indices_of_block_list_2 = []
            # tag_list = []
            for vertical_step_index in range(vertical_block_step):
                for horizontal_step_index in range(horizontal_block_step):
                    indices_of_block_list_2.append([vertical_index * vertical_block_step + vertical_step_index, horizontal_index * horizontal_block_step + horizontal_step_index])
                    # tag_list.append((vertical_index * vertical_block_step + vertical_step_index, horizontal_index * horizontal_block_step + horizontal_step_index))
            indices_of_block_list_2 = np.array(indices_of_block_list_2)
            indices_of_block_list_1.append(indices_of_block_list_2)

    reshaped_precision_list = np.array(precision_list).reshape(-1, int(len(prediction_list) / (configs.col_num * configs.row_num)))
    reshaped_precision_list = np.array([reshaped_precision_list[:, i] for i in range(len(reshaped_precision_list[0]))])
    precision_grids = [reshaped_precision_list[i].reshape(configs.row_num, configs.col_num) for i in range(len(reshaped_precision_list))]
    reshaped_precision_grid = [[[] for _ in range(horizontal_blocks)] for _ in range(vertical_blocks)]

    for block_index in range(len(indices_of_block_list_1)):
        row_index = block_index // horizontal_blocks
        col_index = block_index % horizontal_blocks
        for point_index in range(len(indices_of_block_list_1[block_index])):
            point_vertical_index = indices_of_block_list_1[block_index][point_index][0]
            point_horizontal_index = indices_of_block_list_1[block_index][point_index][1]
            for repetition_index in range(len(precision_grids)):
                reshaped_precision_grid[row_index][col_index].append(precision_grids[repetition_index][point_vertical_index][point_horizontal_index])

    avg_precision_grid = np.mean(reshaped_precision_grid, axis=2)

    print()
    for i in range(len(avg_precision_grid)):
        for j in range(len(avg_precision_grid[i])):
            print(f"{avg_precision_grid[i][j]:.3f}", end="\t")
        print()

    # visualize avg_precision_list
    fig, ax = plt.subplots()
    im = ax.imshow(avg_precision_grid)
    im.set_clim(0.5, 3)
    fig.colorbar(im)
    plt.show()

