import os
import re
import pickle
import streamlit as st
import gdown

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ==========================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_URL = "https://drive.google.com/uc?export=download&id=1c2AClk2_sNoooyr1P9AKkuFvyfzr5tdh"

MODEL_PATH = "sentiment_model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI Model... Please wait..."):
        gdown.download(
            url=MODEL_URL,
            output=MODEL_PATH,
            quiet=False,
            fuzzy=True
        )

TOKENIZER = "tokenizer.pkl"
MAX_LEN = 100
# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    padding-top:20px;
}

.big-font{
    font-size:42px;
    font-weight:bold;
    text-align:center;
    color:#4F8BF9;
}

.small-font{
    text-align:center;
    color:gray;
    font-size:18px;
}

</style>
""",unsafe_allow_html=True)

# ==========================================================
# CHECK FILES
# ==========================================================



if not os.path.exists(TOKENIZER):
    st.error("Tokenizer not found. Run train.py first.")
    st.stop()

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_artifacts():

    model = load_model(MODEL_PATH)

    with open(TOKENIZER,"rb") as f:
        tokenizer = pickle.load(f)

    return model,tokenizer


model,tokenizer = load_artifacts()

# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_text(text):

    text=text.lower()

    text=re.sub(r"<.*?>"," ",text)
    text=re.sub(r"[^a-z ]"," ",text)
    text=re.sub(r"\s+"," ",text)

    return text.strip()

# ==========================================================
# PREDICTION
# ==========================================================

def predict_sentiment(review):

    review=clean_text(review)

    sequence=tokenizer.texts_to_sequences([review])

    sequence=pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probability=float(
        model.predict(sequence,verbose=0)[0][0]
    )

    if probability>=0.5:

        sentiment="😊 Positive"
        confidence=probability*100

    else:

        sentiment="😞 Negative"
        confidence=(1-probability)*100

    return sentiment,confidence

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
'<p class="big-font">🎬 Movie Review Sentiment Analysis</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="small-font">One-to-One Recurrent Neural Network (SimpleRNN)</p>',
unsafe_allow_html=True
)

st.divider()

# ==========================================================
# INPUT
# ==========================================================

review=st.text_area(

    "Enter Movie Review",

    placeholder="Example:\nThis movie was absolutely amazing. Brilliant acting and excellent story.",

    height=220

)

col1,col2=st.columns(2)

with col1:

    predict=st.button(
        "🔍 Predict Sentiment",
        use_container_width=True
    )

with col2:

    clear=st.button(
        "🗑 Clear",
        use_container_width=True
    )

if clear:
    st.rerun()

if predict:

    if review.strip()=="":

        st.warning("Please enter a movie review.")

    else:

        sentiment,confidence=predict_sentiment(review)

        st.divider()

        if "Positive" in sentiment:

            st.success(sentiment)

        else:

            st.error(sentiment)

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

st.divider()

st.subheader("📌 Sample Reviews")

st.info("""
**Positive**

This movie was amazing. Brilliant acting and wonderful storyline.

**Negative**

Worst movie ever. Terrible acting and boring story.
""")

st.divider()

st.caption("Built using TensorFlow • Keras • SimpleRNN • Streamlit")
