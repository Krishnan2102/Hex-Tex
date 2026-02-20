#importing dependencies
import os
import bs4 
from langchain_community.document_loaders import WebBaseLoader
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool


#setting up the environment
os.environ["USER_AGENT"] = "BIT-Mesra-RAG-Project/1.0"
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_API_KEY"]="..."
os.environ["GOOGLE_API_KEY"] = "..."

#setting up model,vector store,embeddings
model = init_chat_model("google_genai:gemini-2.0-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = InMemoryVectorStore(embeddings)

#document loading
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


bs4_strainer=bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only":bs4_strainer},
)
docs=loader.load()

assert len(docs)==1
print(f"Total characters: {len(docs[0].page_content)}")

#document splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

#splitting and storing documents
document_ids = vector_store.add_documents(documents=all_splits)
print(document_ids[:3])
print("Hello")