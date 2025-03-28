from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from deep_translator import GoogleTranslator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los headers
)

# Cargar el modelo entrenado
model = tf.keras.models.load_model("sentiment_analysis_model.h5")

# Cargar el tokenizador usado en el entrenamiento
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Definir la longitud máxima de las secuencias (igual que en el entrenamiento)
max_len = 50

class TextInput(BaseModel):
    text: str

def preprocess_text(text):
    import re
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Eliminar URLs
    text = re.sub(r"@\w+", "", text)  # Eliminar menciones
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Eliminar caracteres especiales
    text = re.sub(r"\s+", " ", text).strip()  # Espacios extra
    return text

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}


@app.post("/predict/")
async def predict_sentiment(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    
    translated_text = GoogleTranslator(source="es", target="en").translate(input_data.text)
    text_clean = preprocess_text(translated_text)
    sequence = tokenizer.texts_to_sequences([text_clean])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="post")
    
    prediction = model.predict(padded_sequence)
    sentiment = np.argmax(prediction)

    sentiments_map = {0: "Negativo", 2: "Positivo"}
    
    return {"text": translated_text, "sentiment": sentiments_map[sentiment]}