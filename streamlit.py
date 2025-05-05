import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Azure SDK
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage as AzureSystemMessage, UserMessage as AzureUserMessage
from azure.core.credentials import AzureKeyCredential

# Pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

# Load environment variables
load_dotenv()

# ----------- Azure Chat LLM Class ------------
class AzureChatLLM(BaseChatModel):
    endpoint: str
    model: str
    token: str

    def __init__(self, **data):
        super().__init__(**data)
        self._client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.token)
        )

    def _generate(self, messages, stop=None, **kwargs) -> ChatResult:
        azure_msgs = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                azure_msgs.append(AzureSystemMessage(content=msg.content))
            elif isinstance(msg, HumanMessage):
                azure_msgs.append(AzureUserMessage(content=msg.content))

        response = self._client.complete(
            messages=azure_msgs,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 1.0),
            model=self.model
        )
        ai_msg = AIMessage(content=response.choices[0].message.content)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    @property
    def _llm_type(self) -> str:
        return "azure-chat"

# ----------- Document Loading and Splitting ------------
def load_and_split(directory):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".csv"):
            loader = CSVLoader(file_path=filepath)
        else:
            continue

        documents = loader.load()
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)

    return all_chunks

# ----------- Vector Store Initialization ------------
def init_vectorstore(chunks):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "test1"

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=index_name,
        embedding=embeddings
    )

# ----------- RAG Chain Construction ------------
@st.cache_resource(show_spinner="Building RAG pipeline...")
def build_rag_chain():
    chunks = load_and_split("Data/")
    vectorstore = init_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    sys_prompt = (
        "You are a friendly and helpful medical assistant. "
        "Answer the user's question in a warm and natural tone, based on the provided information. "
        "Do not mention the source, context, or documents in your response. "
        "If you're unsure, it's okay to say you don't know. "
        "Keep the response short and easy to understand — no more than three sentences.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "{input}")
    ])

    llm = AzureChatLLM(
        endpoint=os.getenv("ENDPOINT"),
        model="openai/gpt-4.1",
        token=os.getenv("GITHUB_TOKEN")
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

# ----------- Streamlit UI ------------
st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🩺")
st.title("🩺 Medical Assistant Chatbot")
st.markdown("Ask a question based on your PDF/CSV documents.")

question = st.text_input("Enter your medical question:")

if question:
    try:
        rag_chain = build_rag_chain()
        with st.spinner("Searching for answers..."):
            result = rag_chain.invoke({"input": question})
            st.success("Answer:")
            st.write(result["answer"])
    except Exception as e:
        st.error(f"Error: {e}")
