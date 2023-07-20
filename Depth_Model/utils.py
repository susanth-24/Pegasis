'''
This module contains the utilities required in training
'''
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth

def Load_Image(path,output_height=480,output_width=640):
    '''
    This function return the pytorch tensor of the image which can later be used for evaluation
    path-> path of the test image, example: "/kaggle/input/nyu-depth-v2/nyu_data/data/nyu2_test/00013_colors.png"
    output_height-> which is set to be 480 for this model
    output_width-> which is set to be 640 for this model
    '''
    image = Image.open(path) 
    transform = transforms.Compose([
        transforms.Resize((output_height, output_width)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  
    # Reshape from (1, 3, 480, 640) to (1, 480, 640, 3)
    image_tensor = image_tensor.permute(0, 2, 3, 1)  
    # Reshape to the required shape (1, 3, 480, 640)
    image_tensor = image_tensor.permute(0, 3, 1, 2)  
    image_tensor = image_tensor.float()

    return image_tensor

def Show_depth(tensor,type="magma"):
    '''
    This will return the depth image,
    matplotlib provide multple types like magma,inferno,viridis,plasma etc, you can change the type accordingly
    '''
    output_reshaped = tensor.squeeze().detach().cpu().numpy()
    im = plt.imshow(output_reshaped, cmap=type)
    return im

