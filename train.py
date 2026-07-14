
import re
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==========================
# CONFIGURATION
# ==========================

DATASET = "IMDB Dataset.csv"

MODEL = "sentiment_model.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 20000
MAX_LEN = 100

# ==========================
# TEXT CLEANING
# ==========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

print("Loading Dataset...")

df = pd.read_csv(DATASET)

print(df.head())
print("Original Shape:", df.shape)

# Shuffle and use first 10k reviews for faster training
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.iloc[:10000]

df["review"] = df["review"].apply(clean_text)
df["sentiment"] = df["sentiment"].map({
    "negative":0,
    "positive":1
})

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(df["review"])

X = tokenizer.texts_to_sequences(df["review"])

X = pad_sequences(
    X,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

y = df["sentiment"].values

with open(TOKENIZER,"wb") as f:
    pickle.dump(tokenizer,f)

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Sequential()

model.add(
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128,
        input_shape=(MAX_LEN,)
    )
)

model.add(
    SimpleRNN(
        128,
        activation="tanh",
        dropout=0.2,
        recurrent_dropout=0.2
    )
)

model.add(Dropout(0.5))

model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL,
    monitor="val_accuracy",
    save_best_only=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ],
    verbose=1
)

loss,accuracy=model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\\nTest Accuracy: {accuracy*100:.2f}%")

pred=(model.predict(X_test)>0.5).astype(int)

print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

with open(TOKENIZER,"wb") as f:
    pickle.dump(tokenizer,f)

print("\\nTraining Completed Successfully!")
print("Saved:")
print(" - sentiment_model.keras")
print(" - tokenizer.pkl")
