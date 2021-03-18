import json

import boto3
from botocore.config import Config

config = Config(
    read_timeout=70,
    retries={
        "max_attempts": 5  # This value can be adjusted to 5 to go up to the 360s max timeout
    },
)
client = boto3.client("sagemaker-runtime", config=config)

request = '{"review_body": "Pathetic design of the caps. Very impractical to use everyday. The caps close so tight that everyday we have to wrestle with the bottle to open the cap. With a baby in one hand opening the cap is a night mare. And on top of these extra ordinary features of super secure cap, they are so expensive when compared to other brands. Stay away from these until they fix the cap issues. We have hurt ourselves many time trying to open caps as they have sharp edges on the inner and outer edges. Not worth the price."}'

response = client.invoke_endpoint(
    EndpointName="aitraining7-ts-hirota",
    ContentType="application/json",
    Accept="application/json",
    Body=request,
)

result = response["Body"]

print(json.load(result))
