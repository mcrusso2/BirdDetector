import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse

from model import resnet18


def load_model(weights_path, class_json, device):
    # Load class names
    with open(class_json, "r") as f:
        class_dict = json.load(f)

    # Reverse mapping: index → class name
    idx_to_class = {int(k): v for k, v in class_dict.items()}

    # Load model
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model, idx_to_class


def predict_image(model, idx_to_class, image_path, device):
    # Same transforms as validation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred_idx = torch.argmax(output, dim=1).item()

    return idx_to_class[pred_idx]


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, idx_to_class = load_model(
        weights_path=args.weights,
        class_json=args.class_json,
        device=device
    )

    prediction = predict_image(model, idx_to_class, args.image, device)

    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str, default="./weights/resnet18_best.pth")
    parser.add_argument("--class-json", type=str, default="./class_indices.json")
    args = parser.parse_args()

    main(args)
