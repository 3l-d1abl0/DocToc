import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()




if __name__ == "__main__":

    
    st.header('DocTalk : Chat with your Docs !')

    st.sidebar.title('LLM, Langchain, chat')
    st.sidebar.markdown('''
        - [streamlit](https://streamlit.io)
    ''')


    #Upload a PDF file
    pdf = st.file_uploader('Upload you pdf file', type='pdf')