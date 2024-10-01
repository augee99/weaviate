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
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    response_body = response['body'].read().decode('utf-8')
    embedding_data = json.loads(response_body)

    return embedding_data['embedding']

# Example query text

query_text = "found it a strangely fraught day . while there had been joy in the reunion with her sister and relief that the rescue had been very timely according to malcolm , who was immensely grateful to have his family brought to a safe place , she was wracked with uncertainty over where she stood with zageo . he had bowed out of any further involvement with her family once they had been brought to the hotel and given accommodation . 'i 'm sure you 'll want some private time together , ' he 'd said , making no appointment with emily for some time alone with him . naturally the moment he had excused himself from their presence , hannah had pounced with a million questions about the sheikh and emily 's involvement with such an unlikely person , given her usual circle of acquaintances . where had she met him ? how long had she known him ? what was their relationship ? why would he do so much for her ? the worst one was-you did n't sell yourself to him , did you , em ? -spoken jokingly , though with a wondering look in her eyes . she had shrugged it off , saying , 'zageo is just very generous by nature . ' 'and drop-dead gorgeous . ' hannah 's eyes had rolled knowingly over what she rightfully assumed was a sexual connection . 'quite a package you 've got there . are you planning on hanging onto him ? ' 'for as long as i can , ' she 'd answered , acutely aware that her time with zageo might well have already ended . a big grin bestowed approval . 'good for you ! not , i imagine , a forever thing , but certainly an experience to chalk up-being with a real life sheikh ! ' not a forever thing ... her sister 's comment kept jangling in her mind . having said good-night to hannah and malcolm and their beautiful little daughters , emily walked slowly along the path to her own accommodation , reflecting on how she had believed her marriage to brian was to be forever . the words-till death do us part-in the marriage service had meant fifty or sixty years down the track , not a fleeting few . it was impossible to know what the future held . life happened . death happened . it seemed to her there were so many random factors involved , it was probably foolish to count on anything staying in place for long . with today 's technology , the world had become smaller , its pace much faster , its boundaries less formidable . even culture gaps were not as wide . or maybe she just wanted to believe that because the thought of being separated from zageo hurt . she wanted more of him . a lot more . on every level . having arrived on the porch"


query_embedding = get_text_embedding_from_bedrock(query_text)

titan4 = client.collections.get("Titan5Collection")
response = titan4.query.near_vector(
    near_vector=query_embedding,  # Your query vector goes here
    limit=2,
    return_metadata=MetadataQuery(distance=True)
)

# Check if any objects were returned
if response.objects:
    for o in response.objects:
        print(o.properties)
        print(o.metadata.distance)
else:
    print("No objects found within the near-vector search.")

response = titan4.query.near_object(
    near_object="a9ae031b-92f1-4128-a8d5-fca06751c053",  # A UUID of an object (e.g. "56b9449e-65db-5df4-887b-0a4773f52aa7")
    limit=2,
    return_metadata=MetadataQuery(distance=True)
)

for o in response.objects:
    print(o.properties)
    print(o.metadata.distance)

client.close()
