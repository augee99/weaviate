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

# Function to search for similar vectors in Weaviate
def search_weaviate(class_name, query_embedding, top_k=1):
    results = client.query.get(
        class_name,
        ["{ _additional { id } properties { file_name chunk_id text } }"]
    ).with_near_vector({
        "vector": query_embedding,
        "certainty": 0.0  # Set a low threshold to see if any results come back
    }).with_limit(top_k).do()

    return results

# Example query text
#query_text = "At three o'clock precisely I was at Baker Street, but Holmes had not yet returned."  # Example query text

query_text = "he did n't want chloe falling into a depression , unable to put it aside to play her part in the show-a very solid reason for her to be here with him , out of her mother 's reach"

# 1. Get the query embedding
query_embedding = get_text_embedding_from_bedrock(query_text)

#print("Query embedding:", query_embedding)

# 2. Search Weaviate for similar vectors
class_name = "Titan4Collection"  # Update with your actual class name
top_k = 5  # Retrieve top 5 results
search_results = search_weaviate(class_name, query_embedding, top_k)

# 3. Process and display the search results
if search_results and "data" in search_results and "Get" in search_results["data"]:
    for obj in search_results["data"]["Get"][class_name]:
        print(f"ID: {obj['_additional']['id']}, File Name: {obj['properties']['file_name']}, Chunk ID: {obj['properties']['chunk_id']}, Text: {obj['properties']['text']}")
else:
    print("No results found.")

all_objects = client.data_object.get(class_name=class_name)

# Print the results
if all_objects and "objects" in all_objects:
    print(f"Found {len(all_objects['objects'])} objects in '{class_name}':")
    for obj in all_objects['objects']:
        print(f"ID: {obj['id']}, Payload: {obj['properties']}")  # Change here to access 'id' directly
else:
    print(f"No objects found in class '{class_name}'.")
