from llama_index.core import SimpleDirectoryReader
print("SimpleDirectoryReader...")
from llama_index.core.node_parser import  SentenceSplitter
print("SentenceSplitter...")
from llama_index.vector_stores.milvus import MilvusVectorStore
print("MilvusVectorStore...")
from pymilvus import MilvusClient
print("MilvusClient...")
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, Settings
print("VectorStoreIndex...")
Settings.llm = None
Settings.chunk_size = 512
Settings.chunk_overlap = 20
print("Settings changed...")


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
print("HuggingFaceLLM...")

documents = SimpleDirectoryReader("./data").load_data()
print("Loaded documents...")

parser = SentenceSplitter()
print("Parser created...")

# nodes = parser.get_nodes_from_documents(documents)
# print("Parsed nodes...")

client = qdrant_client.QdrantClient(
    # you can use :memory: mode for fast and light-weight experiments,
    # it does not require to have Qdrant deployed anywhere
    # but requires qdrant-client >= 1.1.1
    # location=":memory:"
    # otherwise set Qdrant instance address with:
    # url="http://:"
    # otherwise set Qdrant instance with host and port:
    host="localhost",
    port=6333,
    # set API KEY for Qdrant Cloud
    # api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.DXh6V1o09Ok__3IVzadhhipdlRF7sURsZwBQyUVL-0oVHQmwwCRKkPrscROSbvoWsa",
)

print("Qdrant client created...")

vector_store = QdrantVectorStore(client=client, collection_name="samplesearch")
print("Qdrant vector store created...")

storage_context = StorageContext.from_defaults(vector_store=vector_store)
print("Storage context created...")

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model created...")


Settings.embed_model = embed_model
print("Settings updated...",Settings.embed_model)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
print("Index created...")

query_engine = index.as_query_engine()
print("Query engine created...")

response = query_engine.query("Under what circumstances will a court grant a quia timet injunction, and what evidentiary standards must an applicant satisfy to justify such equitable relief?")
print("Query response:", response)

vector_store.persist('./IndexStore')
print("Vector store persisted...")