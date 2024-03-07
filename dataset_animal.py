import torch
import os
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision.transforms import Compose, Resize, ToTensor

class AnimalDataset(Dataset):
    def __init__(self, root, train, transform=None):
        data_path = os.path.join(root, "animals")
        if train:
            data_path = os.path.join(data_path, "train")
        else:
            data_path = os.path.join(data_path, "test")
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            for item in os.listdir(data_files):
                image_path = os.path.join(data_files, item)
                self.image_paths.append(image_path) #k dùng image vì tốn bộ nhớ, nên lưu đường dẫn thôi
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, label

if __name__ == '__main__':
    transformer = Compose([
        ToTensor(),
        Resize((224, 224)),
    ])
    dataset = AnimalDataset("data", train=True, transform=transformer)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    # for images, labels in dataloader:
    #     print(images.shape)
    #     print(labels.shape)