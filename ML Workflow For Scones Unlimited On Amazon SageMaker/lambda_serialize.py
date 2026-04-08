import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    key = event["s3_key"]
    bucket = event["s3_bucket"]
    boto3_object = s3.get_object(Bucket=bucket, Key=key)
    image_data = base64.b64encode(boto3_object["Body"].read()).decode("utf-8")
    return {
        "statusCode": 200,
        "body": {
            "s3_bucket": bucket,
            "s3_key": key,
            "image_data": image_data,
            "inferences": []
        }
    }