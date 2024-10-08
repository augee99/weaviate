import weaviate
import boto3
import json

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
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    response_body = response['body'].read().decode('utf-8')
    embedding_data = json.loads(response_body)

    return embedding_data['embedding']

# Function to search Weaviate
def search_weaviate(class_name, query_text, top_k=5):
    # Get the query embedding
    query_embedding = get_text_embedding_from_bedrock(query_text)
    
    # Perform the search
    results = client.query.get(
        class_name,
        ["{ _additional { id } properties { file_name chunk_id text } }"]
    ).with_near_vector({
        "vector": query_embedding,
        "certainty": 0.0  # Adjust this threshold based on your needs
    }).with_limit(top_k).do()

    return results

# Example query text

query_text = "found it a strangely fraught day . while there had been joy in the reunion with her sister and relief that the rescue had been very timely according to malcolm , who was immensely grateful to have his family brought to a safe place , she was wracked with uncertainty over where she stood with zageo . he had bowed out of any further involvement with her family once they had been brought to the hotel and given accommodation . 'i 'm sure you 'll want some private time together , ' he 'd said , making no appointment with emily for some time alone with him . naturally the moment he had excused himself from their presence , hannah had pounced with a million questions about the sheikh and emily 's involvement with such an unlikely person , given her usual circle of acquaintances . where had she met him ? how long had she known him ? what was their relationship ? why would he do so much for her ? the worst one was-you did n't sell yourself to him , did you , em ? -spoken jokingly , though with a wondering look in her eyes . she had shrugged it off , saying , 'zageo is just very generous by nature . ' 'and drop-dead gorgeous . ' hannah 's eyes had rolled knowingly over what she rightfully assumed was a sexual connection . 'quite a package you 've got there . are you planning on hanging onto him ? ' 'for as long as i can , ' she 'd answered , acutely aware that her time with zageo might well have already ended . a big grin bestowed approval . 'good for you ! not , i imagine , a forever thing , but certainly an experience to chalk up-being with a real life sheikh ! ' not a forever thing ... her sister 's comment kept jangling in her mind . having said good-night to hannah and malcolm and their beautiful little daughters , emily walked slowly along the path to her own accommodation , reflecting on how she had believed her marriage to brian was to be forever . the words-till death do us part-in the marriage service had meant fifty or sixty years down the track , not a fleeting few . it was impossible to know what the future held . life happened . death happened . it seemed to her there were so many random factors involved , it was probably foolish to count on anything staying in place for long . with today 's technology , the world had become smaller , its pace much faster , its boundaries less formidable . even culture gaps were not as wide . or maybe she just wanted to believe that because the thought of being separated from zageo hurt . she wanted more of him . a lot more . on every level . having arrived on the porch"

# Execute the search
search_results = search_weaviate("Titan4Collection", query_text)

# Print the results
print("Search Results:")
if search_results and "data" in search_results and "Get" in search_results["data"]:
    for obj in search_results["data"]["Get"]["Titan4Collection"]:
        print(f"ID: {obj['_additional']['id']}, File Name: {obj['properties']['file_name']}, Chunk ID: {obj['properties']['chunk_id']}, Text: {obj['properties']['text']}")
else:
    print("No results found.")

