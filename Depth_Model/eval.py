import torch
import torch.nn as nn
from utils import Show_depth,Load_Image
from .Depth_model.model import Depth_Model

model=Depth_Model()
torch.save(model.state_dict(), 'add path')
image_tensor=Load_Image(path="add path")

def eval(model,image):
    model.eval()
    with torch.no_grad():
        outputv3 = model(image.to("cuda"))
    Show_depth(outputv3)
#this will show the depth Image
eval(model,image_tensor)

