import weaviate
import boto3
import json

# AWS Bedrock Configuration
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

# Weaviate Configuration
client = weaviate.Client("http://localhost:8080")

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
    return embedding_data['embedding']

# Sample text to insert into Weaviate
sample_text = "Example text for embedding."
sample_embedding = get_text_embedding_from_bedrock(sample_text)

# Insert sample data
#client.data_object.create({
#    "text": sample_text,
#    "file_name": "example.txt",
#    "chunk_id": 1
#}, class_name="Titan4Collection_1", vector=sample_embedding)

# Query Weaviate
query_text = "he 'd give her a puppy or a kitten , something for her to look after and pet , another attraction for staying in the children 's house and it should lessen any loneliness she felt"
query_embedding = get_text_embedding_from_bedrock(query_text)


result = client.query.get("Titan4Collection", ["text", "file_name", "chunk_id"]) \
    .with_near_vector({"vector": query_embedding}) \
    .with_limit(5) \
    .do()

print("Search results:")
print(result)

