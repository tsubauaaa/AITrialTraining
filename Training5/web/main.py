import os
import shutil

import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from werkzeug.utils import secure_filename

from detect import detect

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates/")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


@app.post("/result/")
async def predict_image(request: Request, file: UploadFile = File(...)):
    dir_path = "static/images/upload"
    output_path = "static/images/result"
    # Check the existence and format of the file
    if not file:
        print("File doesn't exist!")
        return templates.TemplateResponse("index.html")
    if not allowed_file(file.filename):
        print(file.filename + ": File not allowed!")
        return templates.TemplateResponse("index.html")

    # save file
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    filename = secure_filename(file.filename)
    filepath = open(os.path.join(dir_path, filename), "wb+")
    fileobj = file.file
    shutil.copyfileobj(fileobj, filepath)

    # Yolo Config Dict
    config = {
        "weights": "weights/last.pt",
        "source": dir_path,
        "output": output_path,
        "img_size": 640,
        "conf_thres": 0.4,
        "iou_thres": 0.5,
        "device": "cpu",
        "view_img": False,
        "save_txt": "store_true",
        "classes": "",
        "agnostic_nms": "store_true",
        "augment": "store_true",
        "update": False,
    }
    # Yolo Detect Objects
    detect(config)
    return templates.TemplateResponse(
        "result.html",
        context={
            "request": request,
            "filepath": "/" + os.path.join(output_path, filename),
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
