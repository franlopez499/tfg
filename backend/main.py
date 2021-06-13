import asyncio
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
import uvicorn
from fastapi import File
from fastapi import Body,FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image

import config
import inference


tags_metadata = [
    {
        "name": "segment",
        "description": "Segment a given image into **COCO** classes.",
    },
    {
        "name": "upscale",
        "description": "Upscale images to super resolution.",
    },
]

app = FastAPI( title="Upscaling and segmentation project",
    description="Documentation for the API that supports super resolution of images and segmentation of objects",
    version="1.0",
    openapi_tags=tags_metadata
    )


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/segment/{label}", tags=["segment"],summary="Performs instance segmentation over an image")
def segment(label: str, bbox: bool = Body(...), score: bool = Body(...), extract: bool = Body(...), file: UploadFile = File(...)):
    """
    Segment an image into **COCO** classes. The image returned will be modified depending on this information:

    - **label**: show labels with the name of the COCO instance detected
    - **bbox**: show bounding box around instances
    - **score**: show the confidence (score) of the detection
    - **extract**: return only the instances detected after applying mask to the image
    - **file**: image to be segmented
    """
    image = np.array(Image.open(file.file))
    start = time.time()
    name = f"/storage/{str(uuid.uuid4())}.jpg"
    print(f"name: {name}")
    inference.segment(name,image, label, bbox,score, extract)

    return {"name": name, "time": time.time() - start}

@app.post("/upscale/{net}/{weight}", tags=["upscale"],summary="Upscale an image to super resolution")
def get_image(net: str, weight:str, file: UploadFile = File(...)):
    """
    Upscale an image using a specified neural network and its weights:

    - **net**: Neural network that will perform the super resolution
    - **weight**: weights with which the neural network performs inference
    - **file**: image to be upscaled
    """    
    image = np.array(Image.open(file.file))
    start = time.time()
    output = inference.upscale(weight, image, net)
    name = f"/storage/{str(uuid.uuid4())}.jpg"
    print(f"name: {name}")
    
    if net == 'RRDN' or net == "RDN":
        output.save(name)
    else:
        cv2.imwrite(name, output)

    return {"name": name, "time": time.time() - start}






if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
