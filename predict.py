from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv('KEY')
endpoint = os.getenv('ENDPOINT')
project_id = os.getenv('PROJECT_ID')
published_name = os.getenv('PUBLISHED_ITERATION_NAME')

credentials = ApiKeyCredentials(in_headers={'Prediction-key':key})
client = CustomVisionPredictionClient(endpoint, credentials)

with open('./images/testimg-data/american-staffordshire-terrier-10.jpg', 'rb') as image:
    results = client.classify_image(project_id, published_name, image.read())

    for prediction in results.predictions:
        print(f'{prediction.tag_name}: {(prediction.probability):.2%}')