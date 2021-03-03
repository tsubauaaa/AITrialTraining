import io
import os

from google.cloud import vision

credential_path = "./aitraining-306004-2e354d0f5ba9.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path

client = vision.ImageAnnotatorClient()

file_name = os.path.abspath("./images/business-card_example.png")

with io.open(file_name, "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)

response = client.text_detection(image=image)
texts = response.text_annotations

for text in texts:
    print('\n"{}"'.format(text.description))
    vertices = [
        "({},{})".format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices
    ]

    print("bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
