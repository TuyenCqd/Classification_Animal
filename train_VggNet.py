import os.path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2 import Compose, Resize, ToTensor, RandomAffine, ColorJitter
from dataset_animal import AnimalDataset
from VggNet import VGGNet
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
import argparse
import shutil
from torchinfo import summary
import matplotlib.pyplot as plt


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def get_args():
    parser = argparse.ArgumentParser(description="Train an CNN model")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--num_workers", "-w", type=int, default=6)
    parser.add_argument("--log_path", "-p", type=str, default="animal_tensorboard")
    parser.add_argument("--checkpoint_path", "-c", type=str, default="animal_checkpoints")
    parser.add_argument("--checkpoint_model", "-m", type=str, default=None)
    parser.add_argument("--lr", "-l", type=float, default=1e-2)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = Compose([
        ToTensor(),
        RandomAffine(degrees=(-5, 5), translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
        Resize((224, 224)),
        ColorJitter(brightness=0.125, contrast=0.25, saturation=0.5, hue=0.05)
    ])
    train_dataset = AnimalDataset(root="data", train=True, transform=train_transform)
    train_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "drop_last": True
    }
    train_dataloader = DataLoader(dataset=train_dataset, **train_params)
    val_transform = Compose([
        ToTensor(),
        Resize((224, 224)),
    ])
    val_dataset = AnimalDataset(root="data", train=False, transform=val_transform)
    val_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "drop_last": False
    }
    val_dataloader = DataLoader(dataset=val_dataset, **val_params)
    model = VGGNet(num_classes=len(train_dataset.categories)).to(device)
    # summary(model, input_size=(1, 3, 224, 224))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.checkpoint_model and os.path.isfile(args.checkpoint_model):
        checkpoint = torch.load(args.checkpoint_model)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_loss = checkpoint["best_loss"]
    else:
        start_epoch = 0
        best_loss = 1000

    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    if os.path.isdir(args.checkpoint_path):
        shutil.rmtree(args.checkpoint_path)
    os.makedirs(args.checkpoint_path)
    writer = SummaryWriter(args.log_path)

    for epoch in range(start_epoch, args.epochs):
        # MODEL TRAINING
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, labels)

            # Backward pass + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description("Epoch: {}/{}. Loss: {:0.4f}".format(epoch + 1, args.epochs, loss.item()))
            writer.add_scalar("Train/loss", loss.item(), iter + epoch * len(train_dataloader))

        # MODEL VALIDATION
        all_losses = []
        all_predictions = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, colour="yellow")
            for iter, (images, labels) in enumerate(progress_bar):
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                predictions = model(images)
                loss = criterion(predictions, labels)
                predictions = torch.argmax(predictions, 1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
                all_losses.append(loss.item())

            acc = accuracy_score(all_labels, all_predictions)
            loss = sum(all_losses) / len(all_losses)
            print("Epoch {}. Validation loss: {}. Validation accuracy: {}".format(epoch + 1, loss, acc))
            writer.add_scalar("Valid/loss", loss, epoch)
            writer.add_scalar("Valid/acc", acc, epoch)
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            plot_confusion_matrix(writer, conf_matrix, train_dataset.categories, epoch)

        # save model
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, "last.pt"))
        if loss < best_loss:
            torch.save(checkpoint, os.path.join(args.checkpoint_path, "best.pt"))
            best_loss = loss


if __name__ == '__main__':
    args = get_args()
    train(args)