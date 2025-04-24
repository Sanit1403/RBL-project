import torch
from PIL import Image
import torchvision.transforms as T
import argparse
import os

# Define class labels (must match training order)
CLASSES = [
    'acanthosis-nigricans',
    'acne',
    'acne-scars',
    'alopecia-areata',
    'dry',
    'melasma',
    'oily',
    'vitiligo',
    'warts'
]

def get_transforms():
    """Returns the transformations applied to input images."""
    return T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ])

def predict(model, img, transform, classes):
    """Predicts the class of a given image using the trained model."""
    model.eval()
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        out = model(img_tensor)
        pred_idx = torch.argmax(out, dim=1).item()
        confidence = torch.softmax(out, dim=1).max().item()

    return classes[pred_idx], confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skin Disease Classification")
    parser.add_argument('-m', '--model', required=True, help="Path to the trained model (.pt)")
    parser.add_argument('-i', '--image', required=True, help="Path to the input image")
    args = parser.parse_args()

    model_path = args.model
    image_path = args.image

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    device = torch.device('cpu')  # Using CPU for simplicity

    # Load model
    model = torch.load(model_path, map_location=device)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms()

    # Make prediction
    predicted_class, confidence = predict(model, image, transform, CLASSES)

    # Output result
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence Score: {confidence:.4f}")
