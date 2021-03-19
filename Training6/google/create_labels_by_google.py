import gc
import glob
import io
import os
import shutil

import tqdm

from google.cloud import vision

credential_path = "./aitraining-306004-2e354d0f5ba9.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path

client = vision.ImageAnnotatorClient()

paths = map(os.path.abspath, glob.glob("./raw_img" + "/*.jp*g", recursive=True))

paths_list = list(filter(os.path.isfile, paths))
paths_list.sort()

del paths

for i in tqdm.tqdm(range(1863, len(paths_list) + 1)):
    file_name = os.path.abspath(paths_list[i])

    with io.open(file_name, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    objects = client.object_localization(image=image).localized_object_annotations

    dir_name = "./vision_api/" + objects[0].name

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    shutil.copy(paths_list[i], dir_name)

    # print(dir_name)
