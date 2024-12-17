# DocuChat AI: Intelligent PDF Conversational Assistant  

Turn your PDFs into interactive chat partners! DocuChat AI is an intelligent PDF conversational assistant that allows you to upload PDFs, ask questions, and receive concise AI-powered answers, all with memory of your chat history.  

## 🚀 Features  

- **PDF Upload**: Upload single or multiple PDFs.  
- **AI-Powered Q&A**: Get quick and accurate answers to your queries based on the uploaded documents.  
- **Chat History**: Context-aware Q&A with memory of previous questions in the current session.  
- **Groq API Integration**: Utilize the Groq `Gemma-7b-It` model for conversational AI.  
- **Text Splitting**: Efficiently handles long documents by splitting into chunks.  

---

## 🛠️ Technologies Used  

- **Streamlit**: Interactive UI for the PDF chatbot.  
- **LangChain**: Framework for language model applications.  
- **HuggingFace Embeddings**: Utilizes the `All-MiniLM-L6-v2` model for embeddings.  
- **FAISS**: Vector store for efficient document retrieval.  
- **Groq API**: Integration with Groq's `Gemma-7b-It` model.  
- **PyPDFLoader**: PDF parsing and content extraction.  

---

## 📦 Installation  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/docuchat-ai.git
   cd docuchat-ai



---

### Notes:
1. Replace `yourusername` in the `git clone` link with your GitHub username.
2. Create a `requirements.txt` with the libraries:
   ```plaintext
   streamlit
   langchain
   langchain_groq
   langchain_huggingface
   langchain_community
   faiss-cpu
   PyPDF2
   transformers

