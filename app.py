import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()




if __name__ == "__main__":

    
    st.header('DocTalk : Chat with your Docs !')

    st.sidebar.title('LLM, Langchain, chat')
    st.sidebar.markdown('''
        - [streamlit](https://streamlit.io)
    ''')

    with st.sidebar:
        # text_input for the OpenAI API key
        api_key = st.text_input('Your OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            st.write(api_key)


    #Upload a PDF file
    pdf = st.file_uploader('Upload you pdf file', type='pdf')

    if pdf:

        if st.button('Start Talking !'):
            st.write('CLICKED')
            st.write(os.environ['OPENAI_API_KEY'])

        st.write(pdf.name)
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        #st.write(text)

        #Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len

        )
        chunks = text_splitter.split_text(text=text)
        st.write(len(chunks))
        st.write(chunks)


    st.subheader('Sourcasdasddddddddddddddddddddddddd')
    st.subheader('Source code')