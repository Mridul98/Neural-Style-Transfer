import torch 
import torch.nn as nn
import torchvision.transforms as tf
from torch.optim import Optimizer
import numpy as np
from PIL import Image
from typing import List,Tuple
from interfaces import CNNModel


class NST:

    def __init__(
        self,
        content_image_path:str,
        style_image_path:str,
        content_layer_index:int,
        style_layer_indices: List[Tuple[int,float]],
        content_weight: float,
        style_weight: float,
        epoch: int,
        cnn_model: CNNModel
        
        ):
        """a class which contains all the functionalities of neural style transfer

        Args:
            content_image_path (str): path to content image
            style_image_path (str): path to style image
            content_layer_index (int): index of the layer from which the feature of
            the content will be extracted
            style_layer_indices (List[Tuple[int,float]]): indices of the layers and its corresponding weights in a tuple
            from which the feature of the styling image will be extracted
            content_weight (float): amount of weight on content loss
            style_weight (float): amount of weight on style loss
            epoch (int): number of epochs to train the NST
            cnn_model (CNNModel): CNN model to be used for neural style transfer
        """

        self.__epoch = epoch
       
        self.__content_weight = content_weight
        self.__style_weight = style_weight

        self.__model = cnn_model(
            content_image=self.__load_img(content_image_path),
            style_image=self.__load_img(style_image_path),
            content_layer_index=content_layer_index,
            style_layer_indices=style_layer_indices  
        )

        self.__content_image = self.__model.content_image
        self.__style_image = self.__model.style_image

        self.__target = torch.randn_like(self.__content_image).requires_grad_(True).cuda()


    def __load_img(self,image_path:str):

        """
        load jpg image
        for now the image is resized to 600 x 400 dimension
        and the pixels are further normalized
        
        returns: torch.Tensor object
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize((600,400))
        im_transform = tf.Compose(
            [
            tf.ToTensor(),
            tf.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
            ]
        )
        img = im_transform(img)
        
        img = torch.unsqueeze(img,0)
        
        return img.cuda()
    
    def __append_image(self, num_iter:int, divider:int):
        """read images from the disc and return a list of image object
        Args:
            num_iter (int): number of iteration. This will be equal to the number
            of epochs that was set to run NST
            divider (int): interval of reading image from the disk.
            if divider=5, it means,pick every 5th imamge from the disk and append 
            it to the image list to be returned

        Returns:
            list[PIL.Image]: a list of PIL.Image object 
        """

        image_list = list()
        for i in range(num_iter):
            if i % divider == 0:
                img = Image.open('seq'+str(i)+'.jpg')
                image_list.append(img)

        return image_list

    def __make_gif(self,imageList:List,name:str):
        """take a list of PIL.Image and make a gif

        Args:
            imageList (List): list of PIL.Image object
            name (str): actual file name of gif file to be
        """

        imageList[0].save(str(name)+'.gif',save_all=True,append_images=imageList[1:],duration=10,loop=0)

    def __gram_matrix(self,tensor:torch.Tensor):
        """calculate gram_matrix

        Args:
            tensor (torch.Tensor): a tensor for which the gram matrix will be 
            calculated

        Returns:
            torch.Tensor: gram matrix of the tensor `tensor`
        """
        batch,channel,width,height = tensor.shape
        tensor = torch.squeeze(tensor)
        tensor = tensor.view(channel,height*width)
        return torch.mm(tensor,tensor.t())

    def __im_convert(self,tensor:torch.Tensor,index:int):
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
        image.save('./results/seq'+str(index)+'.jpg')
  
        return image

    def __get_content_loss(self,target_feature:torch.Tensor,content_feature:torch.Tensor):
        """calculate and return content loss

        Args:
            target_feature (torch.Tensor): target feature
            content_feature (torch.Tensor): content feature

        Returns:
            torch.Tensor: content loss
        """
        
        return torch.mean((target_feature - content_feature)**2)*self.__content_weight

    def __get_style_loss(self,target_style_feature:torch.Tensor,style_gram: List[torch.Tensor],
                        style_gram_weights: List[float]):

        """_summary_

        Returns:
            _type_: _description_
        """
        style_loss = 0

        #TODO: in future, the for loop operation has to be replaced for optimaztion purpose
        for j in range(len(style_gram_weights)):
    
            target_feature = target_style_feature[j]
            target_gram_matrix = self.__gram_matrix(target_feature)
            _, c , h , w = target_feature.shape
            
            layer_style_loss =style_gram_weights[j]*torch.mean((target_gram_matrix - style_gram[j])**2)/(c*h*w)
            style_loss += layer_style_loss

        return self.__style_weight*style_loss
    

    def train(self,optimizer: Optimizer):
        """_summary_

        Args:
            Optimizer (Optimizer): _description_
            style_weight (float): _description_
            content_weight (float): _description_
        """
        print(vars(self.__model))
        content_feature = self.__model.content_features
        style_features = self.__model.style_features
        style_gram = [self.__gram_matrix(style_features[i]) for i in range(len(style_features))]

        optimizer  = optimizer([self.__target],lr = 0.09)

        for i in range(self.__epoch):

            if i % 5 == 0:
                intermediate_image = self.__target.data 
                intermediate_image = self.__im_convert(intermediate_image,i)

            optimizer.zero_grad()

            target_features = self.__model.get_target_features(target_img=self.__target)
            target_features_style = target_features['style']
            target_feature_content = target_features['content']

            content_loss = self.__get_content_loss(
                target_feature=target_feature_content,
                content_feature=content_feature
            )

            style_loss = self.__get_style_loss(
                target_style_feature=target_features_style,
                style_gram=style_gram,
                style_gram_weights=self.__model.style_layer_weight
            )

            total_loss = content_loss + style_loss

            print(f"{i}'th epoch, loss: {total_loss}")

            total_loss.backward(retain_graph = True)
            optimizer.step()


            
