
# Page title
import streamlit as st # type: ignore

#ui design
st.title("VLM Document Classifier with Bert")
st.write("" \
"This project uses a vision language model to classify PDF documents for business transactions occurring within enterprise environments." \
"The model is trained on a custom dataset of business documents, including invoices, purchase orders, shipping orders, inventory, etc."
""
)

#file upload
uploaded_file = st.file_uploader("Choose a PDF file to upload", type=["pdf"])

#uploaded file condition 
#meaning if someone has uploaded a file, then go ahead and do something, if someone does not upload a file, then we can give an error message or any message
if uploaded_file is not None:
    pass 