# ---------------- train_wholeframes.py ----------------

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from model import resnet18


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------- TRANSFORMS -----------
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ]),
    }

    # ----------- LOAD WHOLE FRAMES -----------
    whole_root = os.path.join(args.data_path, "whole_frames")

    train_dataset = datasets.ImageFolder(
        os.path.join(whole_root, "train"),
        transform=data_transform["train"]
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(whole_root, "val"),
        transform=data_transform["val"]
    )

    class_names = train_dataset.classes
    class_num = len(class_names)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ----------- MODEL -----------
    net = resnet18(num_classes=class_num)
    net.load_state_dict(torch.load(args.crops_weights, map_location=device))
    net.to(device)

    # ----------- LOSS + OPTIMIZER -----------
    weight = torch.tensor([1.0, 5.0]).to(device)
    loss_function = nn.CrossEntropyLoss(weight=weight)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    os.makedirs("./weights", exist_ok=True)
    save_path = "./weights/resnet18_final.pth"

    # ----------- TRAIN LOOP -----------
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        # ----------- VALIDATION -----------
        net.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}: val_acc = {val_acc:.4f}")

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
            print("Saved new best whole-frames model")

    print("Phase 2 (whole frames) training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,
                        default="./data/openimages_feeder_optimal")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--crops-weights", type=str,
                        default="./weights/resnet18_crops.pth")
    args = parser.parse_args()

    main(args)
