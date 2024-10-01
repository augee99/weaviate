import weaviate
import boto3
import json

# AWS Bedrock Configuration
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

# Weaviate Configuration
client = weaviate.Client("http://localhost:8080")

Titan4Collection

from weaviate.classes.query import MetadataQuery

all_objects = client.data_object.get(class_name="Titan4Collection")
print(f"Found {len(all_objects['objects'])} objects in 'Titan4Collection':")
for obj in all_objects['objects']:
    print(obj)

