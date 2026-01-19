import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel

# ===============================
# CONFIG STREAMLIT
# ===============================
st.set_page_config(page_title="Vision-LLM FER-CE", layout="centered")
st.title("🧠 Vision-LLM – Reconnaissance d’Émotions Composées")
st.markdown("Upload une image faciale → émotion + explication visuelle (saliency map)")

# ===============================
# DEVICE
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# LOAD MODELS correctement
# ===============================
@st.cache_resource
def load_models():
    # Processor CLIP
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Recréer le modèle CLIPVision
    clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # ⚠️ Autoriser les classes non-sûres pour PyTorch 2.6+
    torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])

    # Charger classifier.pt avec weights_only=False
    classifier = torch.load(
        "notebooks/classifier.pt",
        map_location=device,
        weights_only=False
    )

    clip_vision.to(device).eval()
    classifier.to(device).eval()
    
    return clip_processor, clip_vision, classifier

clip_processor, clip_vision, classifier = load_models()

# ===============================
# EMOTIONS
# ===============================
EMOTION_TEMPLATES = {
    0: 'Happily surprised', 1: 'Happily disgusted', 2: 'Sadly fearful',
    3: 'Sadly angry', 4: 'Sadly surprised', 5: 'Sadly disgusted',
    6: 'Fearfully angry', 7: 'Fearfully surprised',
    8: 'Fearfully disgusted', 9: 'Angrily surprised',
    10: 'Angrily disgusted', 11: 'Disgustedly surprised',
    12: 'Happily fearful', 13: 'Happily sad'
}

# ===============================
# IMAGE UPLOAD
# ===============================
uploaded_file = st.file_uploader("📤 Upload une image faciale", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="Image d'entrée", width=300)

    # ===============================
    # 🔹 PREDICTION
    # ===============================
    inputs = clip_processor(images=[image], return_tensors="pt").to(device)
    features = clip_vision(**inputs).last_hidden_state[:, 0]
    logits = classifier(features)
    probs = torch.softmax(logits, dim=1)

    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item() * 100
    predicted_emotion = EMOTION_TEMPLATES[pred_class]

    st.subheader("🎯 Résultat")
    st.write(f"**Emotion prédite :** {predicted_emotion}")
    st.write(f"**Confidence :** {confidence:.2f}%")

    # ===============================
    # EXPECTED (OPTIONNEL)
    # ===============================
    expected_class = st.selectbox(
        "📋 Classe attendue (pour analyse XAI)",
        list(EMOTION_TEMPLATES.keys()),
        index=5
    )
    expected_emotion = EMOTION_TEMPLATES[expected_class]
    st.write(f"**Expected emotion :** {expected_emotion}")

    # ===============================
    # 🔍 SALIENCY MAP
    # ===============================
    inputs = clip_processor(images=[image], return_tensors="pt").to(device)
    inputs["pixel_values"].requires_grad_()

    features = clip_vision(**inputs).last_hidden_state[:, 0]
    logits = classifier(features)

    classifier.zero_grad()
    logits[0, expected_class].backward()

    grad = inputs["pixel_values"].grad[0].cpu().numpy().transpose(1, 2, 0)
    grad = np.mean(np.abs(grad), axis=2)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

    # ===============================
    # VISUALISATION
    # ===============================
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.array(image))
    ax.imshow(grad, cmap="jet", alpha=0.5)
    ax.set_title(f"Saliency Map – Expected: {expected_emotion}")
    ax.axis("off")

    st.pyplot(fig)
