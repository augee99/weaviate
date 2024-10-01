import boto3
import json
import weaviate

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

# Function to search for the nearest vectors in Weaviate
def search_weaviate(class_name, query_embedding, top_k=1):
    result = client.query.get(class_name) \
        .with_near_vector({"vector": query_embedding}) \
        .with_limit(top_k) \
        .do()

    return result

# Example query text to search
query_text = "At three o'clock precisely I was at Baker Street, but Holmes had not yet returned. The landlady informed me that he had left the house shortly after eight o'clock in the morning"  # Replace with your actual query

# 1. Get the query embedding using AWS Bedrock
query_embedding = get_text_embedding_from_bedrock(query_text)

# 2. Search the Weaviate class for similar vectors
class_name = "Titan4Collection"  # Update with your actual class name in Weaviate
top_k = 3  # Adjust the number of results to retrieve
search_results = search_weaviate(class_name, query_embedding, top_k=top_k)

# 3. Process and display the search results
for result in search_results['data']['Get'][class_name]:
    print(f"ID: {result['_additional']['id']}, Score: {result['_additional']['score']}")

