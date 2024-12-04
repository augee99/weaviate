import weaviate
import os
import uuid
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType


#collection_name = "CohereCollection_1"
collection_name = "CohereMulti_1"
client = weaviate.connect_to_local(
    host="127.0.0.1",  # Use a string to specify the host
    port=8080,
    grpc_port=50051,
)
wvCol = client.collections.get(collection_name)

aggregation = wvCol.aggregate.over_all(total_count=True)
total_count = aggregation.total_count

print("Total number of vectors:", total_count)

client.close()
