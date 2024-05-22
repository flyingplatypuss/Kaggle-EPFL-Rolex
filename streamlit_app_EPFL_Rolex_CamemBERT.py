# This file is a sketch of what the application could have been with the CamemBERT model. Unfortunately, Stramlit does not accept the model due to insufficient space.

import streamlit as st
import requests
from io import BytesIO
import torch
from torch import nn
from transformers import AutoTokenizer, CamembertForSequenceClassification
import spacy

# Load french model
nlp = spacy.load("fr_core_news_md")

class CustomCamembertForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout_prob=0.1):
        super(CustomCamembertForSequenceClassification, self).__init__()
        self.camembert = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.camembert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.camembert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # First token [CLS] for classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return (loss, logits) if loss is not None else (None, logits)

# Load Model and Tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    MODEL_URL = "https://github.com/JohannG3/DS_ML/blob/main/camembert_model_full.pth?raw=true"
    response = requests.get(MODEL_URL)
    model_path = BytesIO(response.content)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()
tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")

def predict_difficulty(sentence):
    tokenized = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**tokenized)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return id2label[prediction]

def get_synonyms(word):
    # Simuler une fonction de récupération de synonymes
    synonyms = {"manger": ["consommer", "dévorer", "ingérer"], "pomme": ["fruit"]}
    return synonyms.get(word, [])

st.title("Prédiction de la difficulté de la langue française avec des suggestions de synonymes")

sentence = st.text_input("Entrez une phrase en français:")

if sentence:
    doc = nlp(sentence)
    prediction = predict_difficulty(sentence)
    st.write(f"Le niveau de difficulté prédit est : {prediction}")

    st.write("Voici quelques synonymes pour les mots de votre phrase :")
    for token in doc:
        syns = get_synonyms(token.text.lower())
        if syns:
            st.write(f"**{token.text}**: {', '.join(syns)}")

    st.write("Essayez une autre phrase pour voir comment elle modifie la difficulté prédite et explorer d'autres synonymes!")

