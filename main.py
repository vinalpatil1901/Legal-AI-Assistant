from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import os
import streamlit as st

load_dotenv()


llm = ChatMistralAI(
    model="mistral-medium",   
    temperature=0.7,
    api_key=os.getenv("MISTRAL_API_KEY")
)


# ------------
# stream lit
st.set_page_config(page_title="LEGAL AI ASSISTANT", layout="wide")
st.title("LEGAL DOCUMENT ASSISTANT")

@st.cache_resource
def load_and_process_docs():
    urls = ["https://indiankanoon.org/doc/11757180/",
        "https://indiankanoon.org/doc/1199182/"
    ]

    documents = []

    for url in urls:
        loader = WebBaseLoader(url)
        documents.extend(loader.load())

    #split
    text_split = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap= 50)
    documents = text_split.split_documents(documents)

    #embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), openai_api_base="https://openrouter.ai/api/v1")
    
    #vector DB
    vectorStore = FAISS.from_documents(documents, embeddings)
    return vectorStore.as_retriever()

retriever = load_and_process_docs()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content= "you are a helpful assistant")]


user_input = st.chat_input("Ask a question about the legal documents:")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    rel_docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in rel_docs])

    prompt = f"""
    YOU ARE A HELPFUL LEGAL ASSISTANT. ANSWER ONLY FROM THE GIVEN CONTEXT:
    
    context:
    {context}

    Question:
    {user_input}
    """
    response = llm.invoke(prompt)

    st.session_state.chat_history.append(AIMessage(content=response.content))

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").write(message.content)






