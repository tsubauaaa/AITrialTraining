import io
import os

import cv2
from google.cloud import vision

credential_path = "./aitraining-306004-2e354d0f5ba9.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path

client = vision.ImageAnnotatorClient()

file_name = os.path.abspath("./images/animal_sample.jpg")

with io.open(file_name, "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)

objects = client.object_localization(image=image).localized_object_annotations

print("Number of objects found: {}".format(len(objects)))
for object in objects:
    print("\n{} (confidence: {})".format(object.name, object.score))
    print("Normalized bounding polygon vertices: ")
    for vertex in object.bounding_poly.normalized_vertices:
        print(" - ({}, {})".format(vertex.x, vertex.y))

img = cv2.imread(file_name)
h, w = img.shape[0:2]
font = cv2.FONT_HERSHEY_DUPLEX
for object in objects:
    rectangles = []
    for vertex in object.bounding_poly.normalized_vertices:
        rectangles.append((vertex.x * w, vertex.y * h))
    cv2.rectangle(img, tuple(map(int, rectangles[0])), tuple(map(int, rectangles[2])), (255, 255, 0), 1)
    cv2.putText(img, object.name, tuple(map(int, rectangles[0])), font, 0.4, (255, 255, 0))
cv2.imwrite("rect.png", img)
