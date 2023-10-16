import random

import numpy as np
import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import configs
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


# def compute_average_picture(pics):
#     avg_pic = np.mean(pics)
#     return avg_pic


class CustomDataset(Dataset):
    @staticmethod
    def custom_sort(class_name):
        row_val, col_val = map(int, class_name.replace("row_", "").replace("col_", "").split('-'))
        return row_val, col_val

    def __init__(self, root_dir, transform=None, train=True, horizontal_step=1, vertical_step=1):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        # 获取所有类别的目录
        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        classes.sort(key=self.custom_sort)

        # 对classes进行分类，row_x相同的为一组。
        grouped_classes = [classes[i:i + configs.col_num] for i in range(0, len(classes), configs.col_num)]
        for i in range(len(grouped_classes)):
            grouped_classes[i] = grouped_classes[i][::horizontal_step]
        grouped_classes = grouped_classes[::vertical_step]
        self.classes = np.array(grouped_classes).reshape(-1)

        self.image_paths = []
        self.labels = []
        for class_name in self.classes:
            row_val, col_val = map(int, class_name.replace("row_", "").replace("col_", "").split('-'))
            if self.train:
                # 如果是训练数据，则选择每个类别的第1张图片
                image_name = os.listdir(os.path.join(root_dir, class_name))[0]
                self.image_paths.append(os.path.join(root_dir, class_name, image_name))
                self.labels.append((row_val, col_val))
            else:
                # 如果是验证数据，则选择每个类别的第4和第7张图片
                image_names = sorted(os.listdir(os.path.join(root_dir, class_name)))
                for image_name in [image_names[3], image_names[6]]:
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



class SimpleCNN(pl.LightningModule):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.validation_outputs = []
        self.training_outputs = []
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


def prepare_data(horizontal_step=1, vertical_step=1):
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(root_dir=f'output/subject_{configs.subject_num}/clipped_{configs.mode}', transform=transform, train=True, horizontal_step=horizontal_step, vertical_step=vertical_step)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = CustomDataset(root_dir=f'output/subject_{configs.subject_num}/clipped_{configs.mode}', transform=transform, train=False, horizontal_step=horizontal_step, vertical_step=vertical_step)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


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

    train_loader, val_loader = prepare_data(1, 1)
    num_epochs = configs.num_epochs

    # 从头训练。
    model = SimpleCNN()
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


def validate_model_of_different_districts(horizontal_step, vertical_step):
    pass


