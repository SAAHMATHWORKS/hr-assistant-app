import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(persist_directory="chroma_db", 
                               chroma_db_impl="duckdb+parquet"))
print("Collections:", client.list_collections())