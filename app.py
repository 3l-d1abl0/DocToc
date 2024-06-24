import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

chunks_size = [0, 0]


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

        chunks_size[0] = st.number_input('Chunk size:', min_value=500, max_value=3000, value=1000)
        chunks_size[1] = st.number_input('Chunk Overlap:', min_value=100, max_value=500, value=200)


    #Upload a PDF file
    pdf = st.file_uploader('Upload you pdf file', type='pdf')

    if pdf:

        if st.button('Start Talking !'):
            st.write('CLICKED')
            st.subheader(chunks_size)

            if 'OPENAI_API_KEY' not in os.environ:
                st.write('Please Provide OpenAPI Key !')
                st.stop()
            else:            
                api_key = os.environ['OPENAI_API_KEY']
                if api_key is None or api_key.strip() == "":
                    st.write('Please Provide OpenAPI Key')
                    st.stop()
                else:

                    #API Key recieved - process further
                    st.write(api_key)   
                    st.write(pdf.name)

                    chunks_size[0] = int(chunks_size[0])
                    chunks_size[1] = int(chunks_size[1])


                    pdf_reader = PdfReader(pdf)
                    text=""
                    for page in pdf_reader.pages:
                        text+=page.extract_text()
                    #st.write(text)

                    #Split the text into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunks_size[0],
                        chunk_overlap=chunks_size[1],
                        length_function=len

                    )
                    chunks = text_splitter.split_text(text=text)
                    #st.write(len(chunks))
                    #st.write(chunks)


                    st.subheader('Sourcasdasddddddddddddddddddddddddd')
                    

                    store_name = pdf.name+'-'+str(chunks_size[0])+'-'+str(chunks_size[1])
                    st.subheader(store_name)

                    #If existing, load embedding from disk, otherwise create
                    if os.path.exists(f"embeddings/{store_name}.pkl"):
                        with open(f"embeddings/{store_name}.pkl", "rb") as f:
                            VectorStore = pickle.load(f)
                        st.write('Embeddings Loaded from the Disk')
                    else:
                        embeddings = OpenAIEmbeddings()
                        VectorStore = FAISS.from_texts(chunks, embeddings)
                        with open(f"embeddings/{store_name}.pkl", "wb") as f:
                            pickle.dump(VectorStore, f)
                        st.subheader('Embeddings Created')

