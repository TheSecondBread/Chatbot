from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

file_path = "chatbot.txt"
loader = TextLoader(file_path)

pages = loader.load_and_split()

print("--------------------pages---------------")
print(pages)

text_splitter = CharacterTextSplitter(separator='\n', chunk_size=400, chunk_overlap=40)

docs = text_splitter.split_documents(pages)

print("----------docs----------------")
print(docs)

# Initialize the embedding function (HuggingFace)
embed_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create the Chroma database from the documents with embeddings
db = Chroma.from_documents(docs, embedding=embed_function, persist_directory="./db")

# Check if the Chroma database was created
print(db)
