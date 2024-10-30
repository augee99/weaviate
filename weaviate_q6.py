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

#query_text = "she set the tray on the glass table and gave herself an imaginary pat on the back for not spilling it , before claiming a seat on the middle of the couch . carrying trays decked with drinks and other liquids was not as easy as it looked . lecie had a newfound respect for those who did it for a living . the morning sun cast its golden hue across the clear , blue sky . lecie stretched past the tray , reaching for her sunglasses on the far side of the table . securing them in the crook of her finger , she dragged them toward her . shielding her eyes behind the shades , she relaxed enough to enjoy the cool morning breeze blowing in from the ocean , bringing with it the scent of the honeysuckle lining the edge of the property . lecie poured herself some orange juice and sat back , sipping it as she took in the view . `` gosh , '' she said out loud , even though she was the only one there . deidra , as far as she knew , was still sleeping , but lecie had brought a glass out for her just in case she happened to wake uncharacteristically early today . lecie gazed out at the ocean beyond the far side of her property . calm and still , it melded together with the early morning blue sky"

#query_text = " walked downstairs and ran my hand along the polished wooden banister , loving the way the old grain felt against the pads of my fingers . this partof town was historic , dating back over one hundred years when confederate soldiers roamed the city"

#query_text = "i had a love for everything old . it was one of the reasons i loved living in virginia . once i became a single mother , i had little time left for myself . my inner nerd had been seriously deprived over the last few years . right now , she was bouncing up and down in excitement . `` so , good surprise ? '' he asked , still seated in the car . `` yes ! perfect . now let 's go ! i want to see everything !'' laughing at my enthusiasm , he opened his door , quickly running around to open mine . he was too late . i was already out of the car , practically foaming at the mouth . i was like a kid in a candy store .my eyes were darting everywhere . there were gardens , an old barn , the house ... i wanted to see it all ! `` i figured a history lover would have visited all the local plantations by now , but i took a chance on this one because of its location and the fact that it was a bed and breakfast . `` it 's magnificent , '' i sighed . it was . whoever owned the property took precious care of it . the pristine gardens had winding paths , budding roses , and ivy covered arches that all lead to a view of the james river that"

query_text = "why was he sad?"

query_embedding = get_text_embedding_from_bedrock(query_text)

cohereCol = client.collections.get("CohereCollection_1")
response = cohereCol.query.near_vector(
    near_vector=query_embedding,  # Your query vector goes here
    limit=4,
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
