import streamlit as st
import json
from utils.model import load_model, preprocess_image
from utils.prediction import predict_disease
from utils.ai import get_ai_response
from utils.definisi import get_definisi_llm
from PIL import Image

# Load model dan class
@st.cache_resource
def load_app_resources():
    model = load_model()
    with open("class_names.json") as f:
        CLASS_NAMES = json.load(f)
    return model, CLASS_NAMES

model, CLASS_NAMES = load_app_resources()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("Fish Disease Classification")
uploaded_file = st.file_uploader("Tambahkan gambar ikan", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Gambar yang diunggah", use_container_width=True)

    image_array = preprocess_image(pil_image)
    predicted_class, confidence, full_pred = predict_disease(model, image_array, CLASS_NAMES)

    st.success(f"Hasil: {predicted_class} ({confidence}%)")

    with st.expander("Probabilitas Lengkap"):
        for i, (label, prob) in enumerate(zip(CLASS_NAMES, full_pred)):
            st.write(f"{i+1}. {label}: {prob:.4f}")
    
    st.subheader(predicted_class)

    if 'current_disease' not in st.session_state or st.session_state.current_disease != predicted_class:
        with st.spinner("Penjelasan Penyakit"):
            st.session_state.current_rekomendasi = get_definisi_llm(predicted_class)
            st.session_state.current_disease = predicted_class
    
    st.info(st.session_state.current_rekomendasi)


    # CHAT SECTION
    st.markdown("---")
    st.subheader("💬 Fish AI")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Tulis pertanyaanmu...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        reply = get_ai_response(st.session_state.chat_history, predicted_class)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
