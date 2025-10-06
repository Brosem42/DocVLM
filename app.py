
# Page title
import streamlit as st # type: ignore
import pypdf #type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification # type: ignore
import torch #type: ignore
import numpy as np # type: ignore
import pickle 
import os
import re
from nltk.corpus import stopwords #type: ignore
from nltk.stem import PorterStemmer #type: ignore
import nltk #type: ignore
from PIL import Image #type: ignore
from pathlib import Path

#how we perform the regular clean text with NLTK + preprocessing techniques 
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


#load fine tuned model + tokenizer
import pickle
def __init__(self, model_path, tokenizer, label_encoder):
    self.model_path = model_path
    self.tokenizer = tokenizer
    self.label_encoder = label_encoder

    absolute_path = os.path.abspath('bert_company_model')
    model_path = AutoModelForSequenceClassification.from_pretrained(absolute_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    return model_path, tokenizer, label_encoder

def clean_text(text): #clean the input text
    text = re.sub(r'[^\w\s]', '', text) #remove special characters
    text = re.sub(r'\s+', ' ', text) #remove extra whitespace
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words]) #removed
    text = text.lower() #convert to lowercase
    return text

model = []
model_path = []
tokenizer = []
label_encoder = []


#function for prediction
def predict(text):
  text = clean_text(text)

  #tokenize the input text
  inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

  #make predicition
  with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits # this is my probability
    predicted_class = torch.argmax(logits, dim=1).item() #this is my maximum class, predicted class
#since I don't want to show a number to my user, i want to show class of my predictor set. so i have to perform inverse transform.
  return label_encoder.inverse_transform([predicted_class])[0]
#PDF extraction functions

def extract_text_from_pdf(file):
    pdf_reader = pypdf.PdfReader(file, strict=False) # type: ignore # create object to read pdf with PdfReader
    text = "" #creating a var to hold the text
    #run looop to extract text from each page because pdfs can have multiple pages + num of pages for user would be unknown, I'll handle it dynamically with loop
    for page in range(len(pdf_reader.pages)): #run pages until end of page (if 3 pages, then the loop will run 3 times)
        text += pdf_reader.pages[page].extract_text() #use built in method from PDF reader, extract text--> loop gives me text
    return text

#ui design
st.title("VLM Document Classifier with Bert")
st.write("" \
"This project uses a vision language model to classify PDF documents for business transactions occurring within enterprise environments. " \
"The model is trained on a custom dataset of business documents, including invoices, purchase orders, shipping orders, inventory, etc." \
""
)

#file upload
uploaded_file = st.file_uploader("Choose a PDF file to upload", type=["pdf"])

#uploaded file condition 
#meaning if someone has uploaded a file, then go ahead and do something, if someone does not upload a file, then we can give an error message or any message
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file) #call the function and extract the text
    
    predicted_class = predict(text)

    st.info(predicted_class)