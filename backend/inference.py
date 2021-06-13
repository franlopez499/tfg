import config
import cv2
from ISR.models import RDN, RRDN
from PIL import Image
import torch
import numpy as np
import RRDBNet_arch as arch
from torchvision import transforms
import os
from collections import OrderedDict
from mrcnn import utils
from mrcnn import model as modellib
from samples import coco
from mrcnn import visualize 
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def create_interpolated_model(alpha):
    net_PSNR_path = './models/RRDB_PSNR_x4.pth'
    net_ESRGAN_path = './models/RRDB_ESRGAN_x4.pth'
    alpha = float(alpha[:3])
    print(alpha) 
    net_interp_path = './models/interp_'+str(int(alpha*10))+'.pth'

    net_PSNR = torch.load(net_PSNR_path)
    net_ESRGAN = torch.load(net_ESRGAN_path)
    net_interp = OrderedDict()
    print('Interpolating with alpha = ', alpha)

    for k, v_PSNR in net_PSNR.items():
        v_ESRGAN = net_ESRGAN[k]
        net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN

    torch.save(net_interp, net_interp_path)
    return net_interp_path


def RRDB_inference(model_path,image):
    device = torch.device('cpu')
    model = arch.RRDBNet(3,3,64,23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    img_LR = transforms.ToTensor()(image).unsqueeze_(0)
    img_LR = img_LR.to(device)
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    return output

def segment(filename,image, label,bbox,score,extract):
    # Load the pre-trained model data
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mrcnn/mask_rcnn_coco.h5")
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
            model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    results = model.detect([image], verbose=1)
    r = results[0]
    classes= r['class_ids']
    print("Total Objects found", len(classes))
    for i in range(len(classes)):
        print(class_names[classes[i]])
    visualize.display_instances(filename, image, label,score,extract,r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'], show_bbox=bbox)
    
    return 

def upscale(weight,image, type):
    model = None
    if type == 'RDN':
        model = RDN(weights=weight)
        sr_img = model.predict(image)
        del model
        return Image.fromarray(sr_img)

    elif type == 'RRDN':
        model = RRDN(weights=weight)
        sr_img = model.predict(image)
        del model
        return Image.fromarray(sr_img)
    elif type == 'RRDB':
        model_path = ''
        if weight == 'ESRGAN':
            model_path = 'models/RRDB_ESRGAN_x4.pth'
        else:
            model_path = 'models/RRDB_PSNR_x4.pth'
        #transforms.ToPILImage()(output)
        return RRDB_inference(model_path,image)  
    else:
        model_path = create_interpolated_model(weight)
        device = torch.device('cpu')
        output = RRDB_inference(model_path,image)
        os.remove(model_path)
        return output


    return 

def inference(model, image):
    model_name = f"{config.MODEL_PATH}{model}.t7"
    model = cv2.dnn.readNetFromTorch(model_name)

    height, width = int(image.shape[0]), int(image.shape[1])
    new_width = int((640 / height) * width)
    resized_image = cv2.resize(image, (new_width, 640), interpolation=cv2.INTER_AREA)

    # Create our blob from the image
    # Then perform a forward pass run of the network
    # The Mean values for the ImageNet training set are R=103.93, G=116.77, B=123.68

    inp_blob = cv2.dnn.blobFromImage(
        resized_image,
        1.0,
        (new_width, 640),
        (103.93, 116.77, 123.68),
        swapRB=False,
        crop=False,
    )

    model.setInput(inp_blob)
    output = model.forward()

    # Reshape the output Tensor,
    # add back the mean substruction,
    # re-order the channels
    output = output.reshape(3, output.shape[2], output.shape[3])
    output[0] += 103.93
    output[1] += 116.77
    output[2] += 123.68

    output = output.transpose(1, 2, 0)
    return output, resized_image
