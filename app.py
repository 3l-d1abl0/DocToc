import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import pickle
#from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
#from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

chunks_size = [0, 0]

app_config ={
    "avatar": { "user": os.path.join(os.path.dirname(os.path.realpath(__file__)), "images", "user-avatar.png"),
                "assistant": os.path.join(os.path.dirname(os.path.realpath(__file__)), "images", "assistant-avatar.png"),
            }
}


def page_setup():
    st.set_page_config(page_title="DocToc", page_icon="🐧")
    st.header('DocToc : Chat with your Docs !')

    st.sidebar.title('LLM, Langchain, chat')
    st.sidebar.markdown('''
        - [streamlit](https://streamlit.io)
    ''')
    st.html(
    """
<style>
    .st-emotion-cache-p4micv {
        width: 2.75rem;
        height: 2.75rem;
    }
</style>
"""
)

def sidebar_setup():
    with st.sidebar:
        # text_input for the OpenAI API key
        api_key = st.text_input('Your OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            st.write(api_key)

        chunks_size[0] = st.number_input('Chunk size:', min_value=500, max_value=3000, value=1000)
        chunks_size[1] = st.number_input('Chunk Overlap:', min_value=100, max_value=500, value=200)

if __name__ == "__main__":

    page_setup()

    sidebar_setup()



    #Upload a PDF file
    pdf = st.file_uploader('Upload you pdf file', type='pdf')

    #If a pdf is added
    if pdf:

        #if "vs" in st.session_state:
            #del st.session_state.vs
            
        #When start talking button is clicked
        if st.button('Start Talking !'):
            
            #st.write('CLICKED')
            #st.subheader(chunks_size)

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
                    

                    store_name = pdf.name+'-'+str(chunks_size[0])+'-'+str(chunks_size[1])
                    #st.subheader(store_name)

                    #If existing, load embedding from disk, otherwise create
                    if os.path.exists(f"embeddings/{store_name}"):
                        #with open(f"embeddings/{store_name}.pkl", "rb") as f:
                            #VectorStore = pickle.load(f)

                        x = FAISS.load_local(f"embeddings/{store_name}", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
                        VectorStore = x.as_retriever()

                        # saving the vector store in the streamlit session state (to be persistent between reruns)
                        st.session_state.vs = VectorStore
                        st.success('Embeddings Loaded from the Disk')
                    else:
                        embeddings = OpenAIEmbeddings()
                        VectorStore = FAISS.from_texts(chunks, embeddings)
                        #with open(f"embeddings/{store_name}.pkl", "wb") as f:
                        #    pickle.dump(VectorStore, f)
                        
                        VectorStore.save_local(f"embeddings/{store_name}")
                        
                        # saving the vector store in the streamlit session state (to be persistent between reruns)
                        st.session_state.vs = VectorStore.as_retriever()
                        st.success('Uploaded, chunked and embedded successfully.')

    if pdf and 'vs' in st.session_state:

        if 'history' not in st.session_state:
            st.session_state['history'] = []
        
        for message in st.session_state['history']:
            role = message["role"]
            avatar_path = (
                app_config["avatar"]["assistant"]
                if role == "assistant"
                else app_config["avatar"]["user"]
            )

            with st.chat_message(role, avatar=str(avatar_path)):
                st.markdown(message["content"])

        if query:= st.text_input("Ask Question from your PDF File"):


            #Add the Prompt by user to Message History
            st.session_state.history.append({"role": "user", "content": query})

            with st.chat_message("user",avatar=str(app_config["avatar"]["user"]),):
                st.markdown(query)
            
            VectorStore = st.session_state.vs
            #docs = VectorStore.similarity_search(query=query, k=3)
            #docs = VectorStore.get_relevant_documents(query)
            docs = VectorStore.invoke(query)

            #retriever
            st.write(VectorStore)
            
            #llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
            # llm = OpenAI(temperature=0.9, max_tokens=500, api_key=OPENAI_API_KEY)
            #initialize_conversation_chain
            llm = ChatOpenAI(model_name="gpt-3.5-turbo")

            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=VectorStore, memory=memory)

            response = conversation_chain({'question': query})


            st.write(response['answer'])

            #st.session_state['history'].append((query, response))
            #Add the Prompt by user to Message History
            st.session_state['history'].append({"role": "assistant", "content": response['answer']})



