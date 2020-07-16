try:
    import unzip_requirements  # noqa
except ImportError:
    pass

import json
import logging
import base64

from imagenet import ImageNetHelper

# Initialize you log configuration using the base class
logging.basicConfig(level=logging.INFO)
# Retrieve the logger instance
logger = logging.getLogger()

def hello(event, context):
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """

headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}
S3_BUCKET = "eva4-p2"

def classify_image(event, context):
    try:
        content_type_header = event["headers"]["content-type"]
        body = base64.b64decode(event["body"])
        logger.info("Body Loaded")

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
    
        img_net = ImageNetHelper("mobilenet_v2") #s3://eva4-2/artifacts/week1/
        img_net.load_model(S3_BUCKET,model_folder="artifacts/week1") 
        picture_tensor = img_net.transform_image(picture.content)
        prediction_idx = img_net.get_prediction(picture_tensor)
        prediction_label = img_net.imagenet1000_classidx_to_label(
            prediction_idx
        )
        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps(
                {"predicted": (prediction_idx, prediction_label)}
            ),
        }
    except Exception as e:
        logger.exception(e)
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": repr(e)}),
        }