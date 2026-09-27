import torch
from transformers import ViTForImageClassification


MODEL_PATH = "best_vit_busi_3class.pth"

CLASS_NAMES = [
    "Benign",
    "Malignant",
    "Normal"
]


def load_model():

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )

    model.load_state_dict(checkpoint)

    model.eval()

    return model