import weaviate
import boto3
import json
import os
import uuid

# AWS Bedrock Configuration
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')  # Specify region and credentials if needed

# Weaviate Configuration
client = weaviate.Client("http://localhost:8080")  # Update with your Weaviate URL if needed

# Function to get embeddings from Amazon Bedrock
def get_text_embedding_from_bedrock(text):
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    response_body = response['body'].read().decode('utf-8')
    embedding_data = json.loads(response_body)

    # Assuming the embedding vector is in the 'embedding' field in the response
    return embedding_data['embedding']

# Create a class in Weaviate for storing the embeddings
def create_weaviate_schema(class_name, vector_size):
    schema = {
        "class": class_name,
        "vectorizer": "none",  # External embeddings are used
        "properties": [
            {
                "name": "file_name",
                "dataType": ["string"]
            },
            {
                "name": "chunk_id",
                "dataType": ["int"]
            },
            {
                "name": "text",
                "dataType": ["string"]
            }
        ]
    }

    if not client.schema.contains({"class": class_name}):
        client.schema.create_class(schema)
        print(f"Class '{class_name}' created successfully.")
    else:
        print(f"Class '{class_name}' already exists.")

def insert_embedding_to_weaviate(collection_name, embedding, payload=None, vector_id=None):
    if vector_id is None:
        # Generate a valid UUID if not provided
        vector_id = str(uuid.uuid4())

    # Define the data object structure for Weaviate
    obj = {
        "properties": payload or {},
        "vector": embedding
    }

    # Create the object in Weaviate
    client.data_object.create(
        data_object=obj,
        class_name=collection_name,  # Weaviate class name (equivalent to Qdrant collection)
        uuid=vector_id  # Optional: can be provided or automatically generated
    )

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
directory_path = '/home/ssm-user/qdrant/data/'  # Update this with the actual directory path containing your files

# Get list of files in the directory (you can filter by extension if needed)
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

# Process a maximum of 10 files
#file_index = 23
for file_index, file_name in enumerate(files[:10]):
    file_path = os.path.join(directory_path, file_name)

    # 1. Read the document from the file
    document_text = read_file(file_path)

    # 2. Chunk the document into smaller parts
    chunk_size = 500  # Adjust the size of chunks as needed
    chunks = list(chunk_text(document_text, chunk_size=chunk_size))

    # 3. Process each chunk
    class_name = "Titan4Collection"  # Update with your actual class name in Weaviate
    embedding_size = 1536

    # Create class in Weaviate if it doesn't exist
    #create_weaviate_schema(class_name, embedding_size)

    # Store objects for insertion
    for idx, chunk in enumerate(chunks):
        embedding = get_text_embedding_from_bedrock(chunk)
        vector_id = file_index * 1000 + idx

        # Insert embedding into Weaviate
        insert_embedding_to_weaviate(
            class_name=class_name,
            embedding=embedding,
            file_name=file_name,
            chunk_id=idx,
            text=chunk,
            vector_id=vector_id
        )

        print(f"Inserted chunk {idx} from file '{file_name}' with ID {vector_id}.")

