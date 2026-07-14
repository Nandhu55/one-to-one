# 🎬 Movie Review Sentiment Analysis using SimpleRNN

A One-to-One Recurrent Neural Network (RNN) project that predicts whether a movie review is **Positive** or **Negative** using TensorFlow, Keras, and Streamlit.

---

## 🚀 Features

- One-to-One RNN Architecture
- Text Preprocessing
- Tokenization & Padding
- SimpleRNN Model
- Streamlit Web Interface
- Confidence Score
- EarlyStopping
- Model Checkpointing

---

## 📂 Project Structure

```
Movie-Review-Sentiment-RNN/

│── app.py
│── train.py
│── requirements.txt
│── IMDB Dataset.csv
│── sentiment_model.keras
│── tokenizer.pkl
│── README.md
```

---

## 📊 Dataset

IMDb Movie Reviews Dataset

Columns

- review
- sentiment

Classes

- Positive
- Negative

---

## 🧠 Model Architecture

```
Input Review
      │
Tokenizer
      │
Padding
      │
Embedding Layer
      │
SimpleRNN
      │
Dense(64)
      │
Dense(32)
      │
Sigmoid Output
      │
Positive / Negative
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/movie-review-rnn.git

cd movie-review-rnn

pip install -r requirements.txt
```

---

## ▶️ Train Model

```bash
python train.py
```

Creates

- sentiment_model.keras
- tokenizer.pkl

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 🧪 Example

Input

```
This movie was amazing. Excellent acting.
```

Output

```
Positive 😊
Confidence: 97%
```

---

## 🛠 Technologies

- Python
- TensorFlow
- Keras
- Streamlit
- Pandas
- NumPy
- Scikit-learn

---

## 👨‍💻 Author

**G. Nandhishwar Reddy**

B.Tech CSE (Data Science)

---

## ⭐ License

This project is created for educational purposes.