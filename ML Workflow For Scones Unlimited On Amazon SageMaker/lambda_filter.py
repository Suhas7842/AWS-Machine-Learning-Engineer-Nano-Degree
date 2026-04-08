import json

THRESHOLD = 0.93

def lambda_handler(event, context):
    
    inferences = event["body"]["inferences"]
    
    meets_threshold = any(i >= THRESHOLD for i in inferences)
    
    if not meets_threshold:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
    
    return event