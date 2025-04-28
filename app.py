import os
from dotenv import load_dotenv
# from fastapibot import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Azure LLM imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage as AzureSystemMessage, UserMessage as AzureUserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# ----- Custom Azure Chat LLM -----
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



#----------------------------------#


# ----- RAG Pipeline Setup --------

# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# def load_and_split(path: str):
#     loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
#     docs = loader.load()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
#     return splitter.split_documents(docs)



def load_and_split(directory):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents = loader.load()

        elif filename.endswith(".csv"):
            loader = CSVLoader(file_path=filepath)
            documents = loader.load()

        else:
            continue  # Skip unsupported file types

        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)

    return all_chunks


#--------vectorstore init------------
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore



def init_vectorstore(chunks):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "test1"
    # if index_name not in pc.list_indexes():
    #     pc.create_index(
    #         name=index_name,
    #         dimension=384,
    #         metric="cosine",
    #         spec=ServerlessSpec(cloud="aws", region="us-east-1")
    #     )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=index_name,
        embedding=embeddings
    )


#--------RAG chain creation------------

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

def build_rag_chain():
    # 1. load and split
    chunks = load_and_split("Data/")
    # 2. init vectorstore
    vectorstore = init_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 3. prompts / llm
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


# Build once
RAG_CHAIN = build_rag_chain()


#----------------------------------#



# ----- FastAPI App -----
app = FastAPI(title="PDF-RAG Chatbot API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# @app.get("/")
# async def root():
#     return {"message": "Hello, world"}


@app.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    try:
        result = RAG_CHAIN.invoke({"input": req.question})
        return QueryResponse(answer=result["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)


#-----------------------------------#

# To run the FastAPI app, use the command:
# uvicorn fastapibot:app --reload
# This will start the server at http://

#-------------curl -----------------#
# curl -X 'POST' \
#   'http://localhost:8000/query' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "question": "What are the symptoms of a fever?"
# }'
