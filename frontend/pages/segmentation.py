import awesome_streamlit as ast
import streamlit as st
from PIL import Image
import time
import requests


LABELS = ['All','BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Segmentation ..."):
        st.title("Instance Segmentation")

        
    st.set_option("deprecation.showfileUploaderEncoding", False)


    image = st.file_uploader("Choose an image")

    label = st.selectbox("Choose the label to segment", [i for i in LABELS])
    bbox = st.checkbox("Bounding Boxes")
    score = st.checkbox("Label with score")
    extract = st.checkbox("Extract instances from the photo")
    if st.button("Segment"):
        if image is not None and label is not None:
            files = {"file": image.getvalue()}
            original = Image.open(image)
            data = {'bbox': bbox, 'score': score, 'extract': extract}
            res = requests.post(f"http://backend:8080/segment/{label}", data=data, files=files)
            img_path = res.json()
            img = Image.open(img_path.get("name"))
            st.write("Tiempo: ", img_path.get("time"))

            st.image(original.resize(size=(original.size[0]*4, original.size[1]*4), resample=Image.BICUBIC),caption="Original photo")
                
            st.image(img,caption="Segmented photo")