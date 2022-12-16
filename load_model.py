import ctypes
import cv2
import numpy as np
# import torch
import logging
logging.basicConfig(level=logging.INFO)

from module.retinaface.load_retinaface import Retinaface_trt
# from module.arcface_torch.backbones import get_model
from module.arcface_paddle import insightface_paddle as face

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='Use execute_async_v2 instead')

from utils import img_warped_preprocess

PLUGIN_LIBRARY = "model/retinaface/mobilenet/libdecodeplugin.so"
engine_file_path = "model/retinaface/mobilenet/retina_mnet.engine"

# name = "r50"
# weight = "model/arcface_torch/backbone.pth"

parser = face.parser()
args = parser.parse_args()
args.output = "output/"
args.use_gpu = True
args.rec = True
args.enable_mkldnn = False
args.rec_model = "ArcFace" #ArcFace
args.index = "datasets/index.bin"

class load_model():
    # RetinaFace_trt
    ctypes.CDLL(PLUGIN_LIBRARY)
    retinaface = Retinaface_trt(engine_file_path)
    retinaface.destroy() # destroy the instance

    # arcface = get_model(name, fp16=False)
    
    # arcface.load_state_dict(torch.load(weight, map_location='cuda:0'))
    # arcface.eval()

    # Arcface_paddle
    arcface = face.InsightFace(args)