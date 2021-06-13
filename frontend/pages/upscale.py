import time
import awesome_streamlit as ast
import requests
import streamlit as st
from PIL import Image

def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Upscale ..."):
        st.title("Upscale images")
    RDN_WEIGHTS = {
        "psnr-large" : "psnr-large",
        "psnr-small" : "psnr-small",
        "noise-cancel" : "noise-cancel",
    }

    RRDN_WEIGHTS = {
        "gans" : "gans",
    }

    RRDB_WEIGHTS = [ "ESRGAN",  "PSNR"]

    NETS = ["RRDN", "RDN", "RRDB", "Interpolate ESRGAN and PSNR nets"]

    st.set_option("deprecation.showfileUploaderEncoding", False)


    image = st.file_uploader("Choose an image")

    net = st.selectbox("Choose the net", [i for i in NETS])

    if net is not None:
        if net == "RRDN": 
            weight = st.selectbox("Choose the type", [i for i in RRDN_WEIGHTS.keys()])
        elif net == "RDN":
            weight = st.selectbox("Choose the type", [i for i in RDN_WEIGHTS.keys()])
        elif net == "RRDB":
            weight = st.selectbox("Choose the type", [i for i in RRDB_WEIGHTS])
        else:
            weight = st.slider('Alpha', min_value=0.0, max_value=1.0,step=0.1)
                



    if st.button("Upscale"):
        if image is not None and weight is not None and net is not None:
            files = {"file": image.getvalue()}
            original = Image.open(image)
            res = requests.post(f"http://backend:8080/upscale/{net}/{weight}", files=files)
            img_path = res.json()
            st.write("Tiempo: ", img_path.get("time"))
            img = Image.open(img_path.get("name"))
            st.image(original.resize(size=(original.size[0]*4, original.size[1]*4), resample=Image.BICUBIC),caption="Original photo")
                
            st.image(img,caption="Improved photo")
                