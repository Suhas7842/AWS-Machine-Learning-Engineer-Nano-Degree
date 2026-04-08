import json
import boto3
import base64

ENDPOINT = "img-classifier-endpoint1"

def lambda_handler(event, context):
    
    body = event["body"]
    image_data = base64.b64decode(body["image_data"])
    
    runtime = boto3.client("runtime.sagemaker")
    
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="image/png",
        Body=image_data
    )
    
    inferences = json.loads(response["Body"].read().decode("utf-8"))
    body["inferences"] = inferences
    
    return {
        "statusCode": 200,
        "body": body
    }