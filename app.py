import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import load_model, CLASS_NAMES


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BUSI ViT Classifier",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}

.prediction-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #f5f7fa;
    text-align: center;
    margin-top: 20px;
}

.prediction {
    font-size: 32px;
    font-weight: 700;
}

.confidence {
    font-size: 20px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🩺 BUSI Breast Ultrasound ViT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Vision Transformer based breast ultrasound image classification'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def get_model():

    model = load_model()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return model, device


model, device = get_model()


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write(
        "**Architecture:** Vision Transformer"
    )

    st.write(
        "**Input Size:** 224 × 224"
    )

    st.write(
        "**Classes:** 3"
    )

    st.write(
        "**Dataset:** BUSI"
    )

    st.divider()

    st.info(
        "Upload a breast ultrasound image "
        "to obtain a model prediction."
    )


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload a breast ultrasound image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # -----------------------------------------------------
    # TWO COLUMNS
    # -----------------------------------------------------

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )


    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    with col1:

        st.subheader(
            "🖼️ Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    with col2:

        st.subheader(
            "🧠 ViT Prediction"
        )

        if st.button(
            "🔍 Analyze Image",
            use_container_width=True
        ):

            image_tensor = transform(
                image
            )

            image_tensor = image_tensor.unsqueeze(
                0
            )

            image_tensor = image_tensor.to(
                device
            )


            # ------------------------------------------------
            # MODEL INFERENCE
            # ------------------------------------------------

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


            predicted_label = CLASS_NAMES[
                predicted_class
            ]


            confidence = probabilities[
                0,
                predicted_class
            ].item() * 100


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="prediction-box">

                <div class="prediction">
                {predicted_label}
                </div>

                <div class="confidence">
                Confidence: {confidence:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            st.subheader(
                "📊 Class Probabilities"
            )

            for class_name, probability in zip(
                CLASS_NAMES,
                probabilities[0]
            ):

                value = probability.item()

                st.write(
                    f"**{class_name}** — "
                    f"{value * 100:.2f}%"
                )

                st.progress(
                    value
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "BUSI ViT deployment | "
    "For research and educational purposes"
)