import weaviate
import boto3
import json
from weaviate.classes.query import MetadataQuery

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
# Example query text


#query_text = "why was he sad?"a

#query_text = "i used to go with natalie , but when she and ben got serious , i declined her invitations to spend the holidays with her family . it felt weird tagging along and i wanted to give them some space . natalie always begged me to join them"

query_text = "i used to go with natalie , but when she and ben got serious , i declined her invitations to spend the holidays with her family . it felt weird tagging along and i wanted to give them some space . natalie always begged me to join them , but i lied and told her i would be fine and already had plans here . with a pang in my chest , i remembered the first time i spent christmas alone . i sat on my couch the whole day , watched a channel that played a christmas story nonstop , and bawled like a baby . after that , i went to the soup kitchen on holidays . i more or less came to terms with not having any family , but the fact that no one except natalie would notice if i died brought on the bout of depression . it was pathetic that i did n't have any other friends that i could spend the holidays with . absolutely pathetic . the serving spoon shook in my hands . i ca n't spend the rest of my life like this . the sting of tears threatened . in a few years i 'll be thirty . the soup kitchen faded away as depression wrapped its coils around my chest like a python , squeezing me of air . i gave up years ago on a happy , picture-perfect life , but it was hard to bear this"

query_embedding = get_text_embedding_from_bedrock(query_text)

cohereCol = client.collections.get("CohereCollection_1")
response = cohereCol.query.hybrid(
    query=query_text, 
    vector=query_embedding,
    limit=3,
    alpha=1,
    return_metadata=MetadataQuery(distance=True,score=True,explain_score=True)
    #return_metadata=MetadataQuery(score=True, explain_score=True)
)

# Check if any objects were returned
if response.objects:
    for o in response.objects:
        print(o.properties)
        #print(o.vector)
        print(f"Score:{o.metadata.score}")
        print(f"Explain:{o.metadata.explain_score}")
        #print(o.metadata.score, o.metadata.explain_score)
else:
    print("No objects found within the near-vector search.")

client.close()
