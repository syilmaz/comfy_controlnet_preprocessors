from ..v1 import openpose_v1, midas 
import cv2
from ..util import resize_image, HWC3, skip_v1
import torch
import numpy as np

def img_np_to_tensor(img_np_list):
    out_list = []
    for img_np in img_np_list:
        out_list.append(torch.from_numpy(img_np.astype(np.float32) / 255.0))
    return torch.stack(out_list)
def img_tensor_to_np(img_tensor):
    img_tensor = img_tensor.clone()
    img_tensor = img_tensor * 255.0
    mask_list = [x.squeeze().numpy().astype(np.uint8) for x in torch.split(img_tensor, 1)]
    return mask_list
    #Thanks ChatGPT

def common_annotator_call(annotator_callback, tensor_image, *args):
    print("img_tensor_to_np 0")
    tensor_image_list = img_tensor_to_np(tensor_image)
    print("img_tensor_to_np 1")
    out_list = []
    out_info_list = []
    for tensor_image in tensor_image_list:
        print("common_annotator_call annotator_callback 1")
        call_result = annotator_callback(resize_image(HWC3(tensor_image)), *args)
        print("common_annotator_call annotator_callback 2")
        H, W, C = tensor_image.shape
        if type(annotator_callback) is openpose_v1.OpenposeDetector:
            out_list.append(cv2.resize(HWC3(call_result[0]), (W, H), interpolation=cv2.INTER_AREA))
            out_info_list.append(call_result[1])
        elif type(annotator_callback) is midas.MidasDetector:
            out_list.append(cv2.resize(HWC3(call_result[0]), (W, H), interpolation=cv2.INTER_AREA))
            out_info_list.append(cv2.resize(HWC3(call_result[1]), (W, H), interpolation=cv2.INTER_AREA))
        else:
            out_list.append(cv2.resize(HWC3(call_result), (W, H), interpolation=cv2.INTER_AREA))
    if type(annotator_callback) is openpose_v1.OpenposeDetector:
        return (out_list, out_info_list)
    elif type(annotator_callback) is midas.MidasDetector:
        return (out_list, out_info_list)
    else:
        print("common_annotator_call annotator_callback 3", out_list)
        return out_list
