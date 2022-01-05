import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as tf
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# GPU
device  = torch.device("cuda: 0 " if torch.cuda.is_available() else "cpu")
print(device)


def load_img(image):
  """
  load jpg image
  for now the image is resized to 600 x 400 dimension
  and the pixels are further normalized
  
  returns: torch.Tensor object
  """
  img = Image.open(image).convert('RGB')
  img = img.resize((600,400))
  im_transform = tf.Compose(
      [
       tf.ToTensor(),
       tf.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
      ]
  )
  img = im_transform(img)
 
  img = torch.unsqueeze(img,0)
  return img

def im_convert(tensor,index):
  """
  get the tensor from the memory of GPU and convert it to
  the jpg image
  """
  image = tensor.to("cpu").clone().detach()
  image = nn.Upsample(scale_factor=2.0,mode='bilinear')(image)
  image = image.numpy().squeeze()
  image = image.transpose(1,2,0)
  image = image * np.array((0.229,0.224,0.225)) + np.array((0.485, 0.456,0.406))
  image = image.clip(0,1)
  image = (image*np.array([255,255,255])).astype(np.uint8)
  image = Image.fromarray(image)
  image.save('seq'+str(index)+'.jpg')
  
  return image

