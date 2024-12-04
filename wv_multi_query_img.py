import weaviate
import boto3
import json
from weaviate.classes.query import MetadataQuery
from PIL import Image
from io import BytesIO
import uuid
import base64

# AWS Bedrock Configuration
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

# Weaviate Configuration
#client = weaviate.Client("http://localhost:8080")

client = weaviate.connect_to_local(
    host="127.0.0.1",  # Use a string to specify the host
    port=8080,
    grpc_port=50051,
)


# Function to get embeddings from Amazon Bedrock
def get_image_embedding_from_bedrock(image):
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

def image_to_base64_data_url(image_path):
  """Converts an image to a base64 data URL."""
  with Image.open(image_path) as img:
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
  return img_base64

query_text = "A girl in a picture"

image_path='/home/ssm-user/weaviate/sample-images/images/image-113.jpg'

img_base64=image_to_base64_data_url(image_path)

query_embedding = get_image_embedding_from_bedrock(img_base64)

titanCol = client.collections.get("CohereMulti_1")
response = titanCol.query.near_vector(
    near_vector=query_embedding,  # Your query vector goes here
    limit=4,
    distance=0.25,
    include_vector=True,
    return_metadata=MetadataQuery(distance=True)
    #return_metadata=MetadataQuery(score=True, explain_score=True)
)

# Check if any objects were returned
if response.objects:
    for o in response.objects:
        print(o.properties)
        #print(o.vector)
        print(o.metadata.distance)
        #print(o.metadata.score, o.metadata.explain_score)
else:
    print("No objects found within the near-vector search.")

client.close()
