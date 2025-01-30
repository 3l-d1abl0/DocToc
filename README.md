# DocToc - Chat with your PDF Documents

DocToc is an interactive document chat application that allows users to have conversations with their PDF documents using OpenAI's language models and Langchain.

## Features

- PDF document upload and processing
- Intelligent text chunking with customizable parameters
- Conversational interface for asking questions about your documents
- Persistent conversation history
- Local storage of embeddings for faster subsequent queries
- Secure API key handling

## Technologies Used

- **Streamlit**: For the web interface
- **LangChain**: For document processing and chat functionality
- **OpenAI**: For embeddings and chat completions
- **FAISS**: For efficient similarity search and vector storage
- **PyPDF2**: For PDF processing

## Prerequisites

- Python 3.x
- OpenAI API key - **You will have to buy some credits from openAI**

## Installation

1. Clone the repository
2. Create a virtual env
```
$ python -m venv <virtual-environment-name>
```
3. Install the required dependencies:
```
$ source <virtual-environment-name>/bin/activate
$ pip install -r requirements.txt
```
4. Run your application:
```
$ streamlit run app.py
```
5. Enter your OpenAI API key in the sidebar
6. Upload a PDF document
7. Configure chunk size and overlap parameters (optional)
8. Click "Start Talking!"
9. Begin asking questions about your document

## Configuration Options

- **Chunk Size**: Controls the size of text segments (500-3000 characters)
- **Chunk Overlap**: Controls the overlap between segments (100-500 characters)

## Features in Detail

### Document Processing
- Converts PDF to text
- Splits text into manageable chunks
- Creates and stores embeddings locally

### Conversation
- Maintains chat history
- Uses GPT-3.5-turbo for responses
- Retrieves relevant context for each query

## Security Note

The application handles the OpenAI API key securely through password-protected input.

## Contributing

Feel free to submit issues and enhancement requests!

---
Built with ❤️ using Streamlit and LangChain
