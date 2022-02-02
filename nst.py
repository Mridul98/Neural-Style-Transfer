import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as tf
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

imageList = list()


# GPU
# device  = torch.device("cuda: 0 " if torch.cuda.is_available() else "cpu")
# print(device)

def appendImage(num_iter,divider):
  for i in range(num_iter):
    if i % divider == 0:
      img = Image.open('seq'+str(i)+'.jpg')
      imageList.append(img)

def make_gif(imageList,name):
    imageList[0].save(str(name)+'.gif',save_all=True,append_images=imageList[1:],duration=10,loop=0)

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

def saveIamge(tensor):
  image = tensor.cpu().clone().detach()
  image = torch.squeeze(image).transpose(1,2,0).numpy()
  image = image*np.array((0.229,0.224,0.225))+np.array((0.485,0.456,0.406))
  image = image.clip(0,1)
  image = tf.ToPILImage()(image)
  image.save('/content/art.jpg')

def gram_matrix(tensor):
    batch,channel,width,height = tensor.shape
    tensor = torch.squeeze(tensor)
    tensor = tensor.view(channel,height*width)
    return torch.mm(tensor,tensor.t())

def get_feature(image,model,indexes):
  feature = list()
  x = image
  for i , layer in enumerate(model.features):
    x = layer(x)
    if i in indexes:
      
      feature.append(x)
  return feature

vgg_19 = models.vgg19(pretrained=True)
for params in vgg_19.parameters():
  params.requires_grad = False
vgg_19.cuda().eval()
for i, layer in enumerate(vgg_19.features):
  if isinstance(layer, torch.nn.MaxPool2d):
    vgg_19.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

content_layer_index = 22
style_layer_indexes = [ 0 ,5, 10, 19, 25, 34 ]
style_layer_indexes.append(content_layer_index)
indexes = sorted(style_layer_indexes)

contentImage = load_img('content.jpg')
styleImage = load_img('style.jpg')
contentImage , styleImage = contentImage.cuda() , styleImage.cuda()
contentFeatures = get_feature(contentImage,vgg_19,indexes)[4]
style_features = get_feature(styleImage,vgg_19,indexes)
del style_features[4]

target = torch.randn_like(contentImage).requires_grad_(True).cuda()
optimizer  = optim.Adam([target],lr = 0.09)

def train(epoch,contentWeight,styleWeight):
  for i in range(epoch):
     if i % 5 == 0:
       intermediateImage = target.data
       intermediateImage = im_convert(intermediateImage,i)
       
     optimizer.zero_grad()
     target_features = get_feature(target, vgg_19,indexes)
     _ , ct,ht,wt = target_features[4].shape
     content_loss = torch.mean((target_features[4] - contentFeatures)**2)
     del target_features[4]
     style_loss = 0
 
     for j in range(len(style_gram_weights)):
 
       target_feature = target_features[j]
       target_gram_matrix = gram_matrix(target_feature)
       _, c , h , w = target_feature.shape
       
       layer_style_loss =style_gram_weights[j]*torch.mean((target_gram_matrix - style_gram[j])**2)/(c*h*w)
       style_loss += layer_style_loss
     totalLoss = contentWeight*content_loss + styleWeight*style_loss
     print(i , totalLoss.data)
     totalLoss.backward(retain_graph = True)
     optimizer.step()


content_layer_index = 22
style_layer_indexes = [ 0 ,5, 10, 19, 25, 34 ]
style_gram_weights = [0.75,0.5,0.5,0.3,0.3,0.3]
style_layer_indexes.append(content_layer_index)
indexes = sorted(style_layer_indexes)


style_gram = [gram_matrix(style_features[i]) for i in range(len(style_features))]


contentImage = load_img('content.jpg')
styleImage = load_img('style.jpg')
contentImage , styleImage = contentImage.cuda() , styleImage.cuda()
contentFeatures = get_feature(contentImage,vgg_19,indexes)[4]
style_features = get_feature(styleImage,vgg_19,indexes)
del style_features[4]


if __name__ == '__main__':

  train(200,1e4,1e2)
  appendImage(200,5)
  make_gif(imageList,'output')