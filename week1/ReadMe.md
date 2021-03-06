# Session 1 - Deploying Mobilenet_v2 model over AWS

Here is group GitHub page https://github.com/EVA4-RS-Group/

### **Objective**:

Deploy a pre-trained  model of Mobilenet_v2 on AWS Lambda using Serverless and classify  images.

### **Results**:

1. Here is group endpoint https://fyia7867vd.execute-api.ap-south-1.amazonaws.com/dev/classify
2. Hosted at my endpoint : https://pa1f737kg4.execute-api.us-east-1.amazonaws.com/dev/week1-dev-classify_image
3. Input image : data/Yellow-Labrador-Retriever
4. Response image screenshot: api_response_insomnia_snapshots.png

## 2. Steps (Developer Section)

#### Step 0 Pre-requisites

1. Confirm docker is present on the machine
2. Confirm Node.js is present on the machine.
3. Confirm python3.8 is on the machine
4. Sign-up for AWS  https://aws.amazon.com/free/, Create IAM account and get its access key and secret key 
5. Install the Serverless framework: sudo npm install -g serverless
6. Ensure that AWS credentials are configured

#### Step 1 Download model

1. Download a pre-trained MobileNet_v2 model using code present get_models-to_s3.py
2. create an S3 bucket, which holds the model add a Pytorch to the Lambda Environment
3. Ensure that IAM user account can write and read object from S#

#### Step 2 Write and Deploy AWS Lambda

1. Create a Python Lambda function with serverless Framework
   -- creates handler.py and serverless.xml
2. Install python requirements plugin, adding requirements.txt
   --serverless plugin install -n serverless-python-requirements
3. write a prediction function to classify an image inside handler.py
4. Ensure print statements are added in the code. These statement get logged in AWS Watchlog for debugging purposes.
5. Configure the serverless framework to set up the API gateway for inference. serverless.yml
6. Deploy AWS lamba function using serverless framework
7. Ensure that media types multipart/form-data and */* are added in AWS API Gateway
8. Test our deployment using Insomnia
   --api, POST with multipart/form-data

## 3.References 

https://towardsdatascience.com/scaling-machine-learning-from-zero-to-hero-d63796442526