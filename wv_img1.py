import weaviate
import boto3
import json
import os
from PIL import Image
from io import BytesIO
import uuid
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
import base64

# AWS Bedrock Configuration
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')  # Specify region and credentials if needed

# Weaviate Configuration
#client = weaviate.Client("http://localhost:8080")  # Update with your Weaviate URL if needed

collection_name = "CohereMulti_1"
client = weaviate.connect_to_local(
    host="127.0.0.1",  # Use a string to specify the host
    port=8080,
    grpc_port=50051,
)
wvCol = client.collections.get(collection_name)
client.collections.delete(collection_name)

def create_weaviate_collection(client,collection_name):
    client.collections.create(
        collection_name,
    #`    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        properties=[  # properties configuration is optional
            Property(name="file_name", data_type=DataType.TEXT),
            Property(name="chunck_id", data_type=DataType.INT),
            Property(name="image_name", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="image", data_type=DataType.BLOB)
        ]
    )

# Function to get embeddings from Amazon Bedrock
def get_image1_embedding_from_bedrock(image):
    output_embedding_length=256
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-image-v1",
        #modelId="cohere.embed-english-v3",
        contentType="application/json",
        accept="*/*",
        #body = {"inputImage": [image]}
        body = json.dumps({"inputImage": image, "embeddingConfig": {"outputEmbeddingLength": output_embedding_length}})
    )
    
    response_body = json.loads(response.get('body').read())
    return response_body["embedding"]

# Function to get embeddings from Amazon Bedrock
def get_text_embedding_from_bedrock(text):
    output_embedding_length=256
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-image-v1",
        #modelId="cohere.embed-english-v3",
        contentType="application/json",
        accept="*/*",
        #body = {"inputImage": [image]}
        body = json.dumps({"inputText": text, "embeddingConfig": {"outputEmbeddingLength": output_embedding_length}})
    )
    
    response_body = json.loads(response.get('body').read())
    return response_body["embedding"]


# Function to get image embeddings from Amazon Bedrock
def get_image_embedding_from_bedrock(processed_image):
    response = bedrock_client.invoke_model(
        modelId="cohere.embed-english-v3",
        images=[processed_image],
        input_type='image',
        embedding_types=["float"]
    )
    return response.embeddings.float[0]

# Function to read the content from a file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to chunk text into smaller parts
def chunk_text(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])


def image_to_base64_data_url(image_path):
  """Converts an image to a base64 data URL."""
  with Image.open(image_path) as img:
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
  return img_base64

# Directory path containing the files
text_path = '/home/ssm-user/weaviate/data/'  #date this with the actual directory path containing your files
img_path = '/home/ssm-user/weaviate/sample-images/images/' #Sample images

# Get list of files in the directory (you can filter by extension if needed)
files = [f for f in os.listdir(text_path) if os.path.isfile(os.path.join(text_path, f))]

img_files = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]

create_weaviate_collection(client,collection_name)

# Process images
for file_index,file_name in enumerate(img_files):
    file_path = os.path.join(img_path, file_name)

    # 1. Read the document from the file
    img_data = image_to_base64_data_url(file_path)
    processed_data = f"data:image/png;base64,{img_data}"
    
    embedding = get_image1_embedding_from_bedrock(img_data)
    print(embedding)
    uuid = wvCol.data.insert(
        properties={
            "image_name": file_name,
            "image": img_data
        },
        vector=embedding
    )
    print(f"Inserted mage file '{file_name}' with ID {uuid}.")

#Process a maximum of 10 files
for file_index, file_name in enumerate(files[:10]):
    file_path = os.path.join(text_path, file_name)

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
