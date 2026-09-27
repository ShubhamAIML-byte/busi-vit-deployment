import torch
from PIL import Image
from torchvision import transforms
from model import load_model, CLASS_NAMES


# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("Loading trained ViT model...")

model = load_model()
model.to(device)
model.eval()

print("Model loaded successfully!")


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ---------------------------------------------------------
# IMAGE PATH
# ---------------------------------------------------------

image_path = input(
    "\nEnter the path of your test image: "
)


# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------

image = Image.open(image_path).convert("RGB")

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

with torch.no_grad():

    outputs = model(
        pixel_values=image_tensor
    )

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

predicted_label = CLASS_NAMES[predicted_class]

confidence = probabilities[
    0,
    predicted_class
].item() * 100


print("\n" + "=" * 50)

print("BUSI ViT PREDICTION")

print("=" * 50)

print(
    f"Prediction : {predicted_label}"
)

print(
    f"Confidence : {confidence:.2f}%"
)

print("\nClass probabilities:")

for class_name, probability in zip(
    CLASS_NAMES,
    probabilities[0]
):

    print(
        f"{class_name:12s}: "
        f"{probability.item() * 100:.2f}%"
    )

print("=" * 50)