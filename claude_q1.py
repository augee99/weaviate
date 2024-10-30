import weaviate
import boto3
import json
from weaviate.classes.query import MetadataQuery


# AWS Bedrock Configuration
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

weaviate_client = weaviate.connect_to_local(
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

def find_context_text_from_weaviate(text):

    query_embedding = get_text_embedding_from_bedrock(query_text)
    cohereCol = weaviate_client.collections.get("CohereCollection_1")
    response = cohereCol.query.near_vector(
        near_vector=query_embedding,  # Your query vector goes here
        limit=3,
        include_vector=True,
        return_metadata=MetadataQuery(distance=True)
    )
    text_values = []
    if response.objects:
        for o in response.objects:
            text_value = o.properties.get("text", "")
            text_values.append(text_value)  # Add the text_value to the list
            #print(text_values)
            return text_values
    else:
        print("No objects found within the near-vector search.")


def generate_cluade_response(prompt, context_text):

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "top_k": 250,
        "stop_sequences": [],
        "temperature": 1,
        "top_p": 0.999,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{context_text}\n\n{prompt}"
                    }
                ]
            }
        ]
    }

    response = bedrock_client.invoke_model(
        #modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        modelId="arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0",
        body=json.dumps(body),
        contentType='application/json',
        accept='application/json'
    )

    return json.loads(response['body'].read().decode('utf-8'))

#query_text = "i had a love for everything old . it was one of the reasons i loved living in virginia . once i became a single mother , i had little time left for myself . my inner nerd had been seriously deprived over the last few years . right now , she was bouncing up and down in excitement . `` so , good surprise ? '' he asked , still seated in the car . `` yes ! perfect . now let 's go ! i want to see everything !'' laughing at my enthusiasm , he opened his door , quickly running around to open mine . he was too late . i was already out of the car , practically foaming at the mouth . i was like a kid in a candy store .my eyes were darting everywhere . there were gardens , an old barn , the house ... i wanted to see it all ! `` i figured a history lover would have visited all the local plantations by now , but i took a chance on this one because of its location and the fact that it was a bed and breakfast . `` it 's magnificent , '' i sighed . it was . whoever owned the property took precious care of it . the pristine gardens had winding paths , budding roses , and ivy covered arches that all lead to a view of the james river that"

#query_text = "single mother"
query_text = "why was he sad?"

context_text = find_context_text_from_weaviate(query_text)
#prompt = "What's the store refered in the context ?"
prompt = "summarize the conext"

response = generate_cluade_response(prompt, context_text)
text_value = response['content'][0]['text']
print(text_value)

weaviate_client.close()

