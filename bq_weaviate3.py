import weaviate
import boto3
import json
import os
import uuid
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType

# AWS Bedrock Configuration
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')  # Specify region and credentials if needed

# Weaviate Configuration
#client = weaviate.Client("http://localhost:8080")  # Update with your Weaviate URL if needed

collection_name = "CohereCollection_1"
client = weaviate.connect_to_local(
    host="127.0.0.1",  # Use a string to specify the host
    port=8080,
    grpc_port=50051,
)
wvCol = client.collections.get(collection_name)
client.collections.delete(collection_name)
client.collections.create(
    collection_name,
#`    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    properties=[  # properties configuration is optional
        Property(name="file_name", data_type=DataType.TEXT),
        Property(name="chunck_id", data_type=DataType.INT),
        Property(name="text", data_type=DataType.TEXT)
    ]
)
# Function to get embeddings from Amazon Bedrock
def get_text_embedding_from_bedrock(text):
    response = bedrock_client.invoke_model(
        #modelId="amazon.titan-embed-text-v1",
        modelId="cohere.embed-english-v3",
        contentType="application/json",
        accept="*/*",
        body = json.dumps({"texts": [text], "input_type": "search_document"})
    )
    
    response_body = json.loads(response.get('body').read())
    embeddings = []
    for i, embedding in enumerate(response_body.get('embeddings')):
        embeddings.append(embedding)
    
    #print(embeddings)
    return embeddings[0]

# Function to read the content from a file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to chunk text into smaller parts
def chunk_text(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

# Directory path containing the files
directory_path = '/home/ssm-user/weaviate/data/'  # Update this with the actual directory path containing your files

# Get list of files in the directory (you can filter by extension if needed)
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

# Process a maximum of 10 files
for file_index, file_name in enumerate(files[:10]):
    file_path = os.path.join(directory_path, file_name)

    # 1. Read the document from the file
    document_text = read_file(file_path)

    # 2. Chunk the document into smaller parts
    chunk_size = 250  # Adjust the size of chunks as needed
    chunks = list(chunk_text(document_text, chunk_size=chunk_size))
   
    for idx, chunk in enumerate(chunks):
        embedding = get_text_embedding_from_bedrock(chunk)
        #vector_id = file_index * 1000 + idx
        uuid = wvCol.data.insert(
            properties={
                "file_name": file_name,
                "chunk_id": idx,
                "text": chunk
            },
            vector=embedding
        )

        print(f"Inserted chunk {idx} from file '{file_name}' with ID {uuid}.")

client.close()
