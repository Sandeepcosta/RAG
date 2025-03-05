import os
import numpy as np
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import PyPDF2

# ----------------------------- CONFIGURATION --------------------------------

# Elasticsearch Configuration
ELASTICSEARCH_URL = "http://11.0.0.145:9200/"
INDEX_NAME = "rag_index_test"

# IBM Watson API Configuration
IBM_API_KEY = "qdyGtVucbt6PtoHyk5QAnFqjeat4WrbgSDcCPfk3VAyn"
IBM_PROJECT_ID = "2f606c5f-f67f-4397-8635-6ca6358f8440"
WATSON_URL = "https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/your-instance-id"

# PDF File Path
PDF_FILE_PATH = "/home/sandeep-cc/Documents/rag/documents/Elastic_NV_Annual-Report-Fiscal-Year-2023.pdf"

# ----------------------------- INITIALIZATION --------------------------------

# Initialize Elasticsearch client
es = Elasticsearch(ELASTICSEARCH_URL)

# Check if Elasticsearch is accessible
if not es.ping():
    raise Exception("Cannot connect to Elasticsearch. Check the URL and connection.")

# Initialize Sentence Transformer Model
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Initialize IBM Watson NLP
authenticator = IAMAuthenticator(IBM_API_KEY)
nlu = NaturalLanguageUnderstandingV1(version="2021-08-01", authenticator=authenticator)
nlu.set_service_url(WATSON_URL)

# ----------------------------- INDEX CREATION --------------------------------

def create_index(index_name):
    """Creates an Elasticsearch index with dense_vector mapping."""
    index_mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384  # Ensure this matches the embedding model dimensions
                }
            }
        }
    }
    es.indices.create(index=index_name, body=index_mapping, ignore=400)
    print(f"Index '{index_name}' created successfully.")

create_index(INDEX_NAME)

# ----------------------------- PDF TEXT EXTRACTION --------------------------------

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Extract text from the PDF
document_text = extract_text_from_pdf(PDF_FILE_PATH)

# ----------------------------- DOCUMENT INDEXING --------------------------------

def index_document(text, index_name=INDEX_NAME):
    """Indexes a document with its text and embedding in Elasticsearch."""
    embedding = embedding_model.encode(text, convert_to_numpy=True)
    doc_body = {"text": text, "embedding": embedding.tolist()}
    es.index(index=index_name, body=doc_body)
    print("Document indexed successfully.")

# Index extracted text
index_document(document_text)

# ----------------------------- DOCUMENT RETRIEVAL --------------------------------

def retrieve_documents(query, k=3, index_name=INDEX_NAME):
    """Retrieves top-k relevant documents from Elasticsearch using k-NN search."""
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    query_vector = np.array(query_embedding).tolist()

    response = es.search(
        index=index_name,
        body={
            "query": {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": k
                }
            }
        }
    )
    
    return [hit["_source"]["text"] for hit in response["hits"]["hits"]]

# Example Query
query = "who audits Elastic?"
retrieved_docs = retrieve_documents(query)
print("Retrieved Documents:", retrieved_docs)

# ----------------------------- IBM WATSON NLP ANALYSIS --------------------------------

def generate_response_with_ibm_watson(query, retrieved_docs):
    """Uses IBM Watson NLP to analyze retrieved documents and generate a response."""
    input_text = query + "\n" + "\n".join(retrieved_docs)

    response = nlu.analyze(
        text=input_text,
        features={
            "entities": {},
            "keywords": {},
            "sentiment": {},
            "categories": {}
        }
    ).get_result()

    sentiment = response.get("sentiment", {}).get("document", {}).get("label", "Neutral")
    keywords = [keyword["text"] for keyword in response.get("keywords", [])]

    return f"Sentiment: {sentiment}\nKeywords: {', '.join(keywords)}\n"

# Generate Response from IBM Watson
response_text = generate_response_with_ibm_watson(query, retrieved_docs)
print("Generated Response from Watson NLP:", response_text)
