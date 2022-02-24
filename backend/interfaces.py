from abc import ABC, abstractmethod
from time import time
from typing import List, Union
import torch
import torchvision.models




class GifInterface(ABC):
    """abstract interface for making 
    gif from the list of images
    """
    def append_images():
        pass
    def make_gif():
        pass

class Calculatable(ABC):

    @abstractmethod
    def calculate(self):
        pass

class FeatureExtractor(ABC):

    @abstractmethod
    def __init__(self,img_tensor:torch.Tensor, pretrained_model: torch.nn.Module) -> None:
        """abstract class that should be extended for extracting features
        from the modified CNN model 

        Args:
            img_tensor (torch.Tensor): image tensor for which the feature will be extracted 
            from the pretrained modified CNN model
            pretrained_model (torch.nn.Module): modified CNN model
        """
        super().__init__()

    @abstractmethod
    def get_features(self,layer_indices: Union[List[int],int]):
        """Returns a list of features extracted from specified layers of the 
        pretrained model by passing img_tensor through the model layers
        """
        pass

class CNNModel(ABC):
    """abstract interface for CNN model
    to be used for Neural Style Transfer
    """
    @abstractmethod
    def __init__(self):
        pass
    
    @property
    @abstractmethod
    def style_features(self):
        """get style features
        """
        pass
    
    @property
    @abstractmethod
    def content_features(self):
        """get content features"""
        pass

    @property
    @abstractmethod
    def content_layer_index(self):
        """
        index of the layer from which the feature of
        the content will be extracted
        """
        pass

    @property
    @abstractmethod
    def style_layer_indices(self):
        """
        indices of the layers from which the feature of
        the styling image will be extracted
        """
        pass

    @property
    @abstractmethod
    def style_layer_weight(self):
        """weight of the style layers 
        """
        pass

    @style_layer_weight.setter
    def style_layer_weight(self,weight_list:float):
        """set the weight from the specified list """
        pass

    @content_layer_index.setter
    def content_layer_index(self,index_list: int):
        """set the layer from which the feature of 
        content image will be extracted

        Args:
            layer_index (int): index of CNN layer from 
            which the feature will be extracted from
        """
        pass
    
    @style_layer_indices.setter
    def style_layer_indices(self,index_list: List[int]):
        """set the weight for 

        Args:
            layer_indices (List[int]): _description_
        """
        pass
    
    @property
    @abstractmethod
    def model(self):
        """
            cnn model for feature extraction 
        """
        pass
    
    @model.setter
    @abstractmethod
    def model(self,pretrained_model:torch.nn.Module):
        """subclass must implement this setter method of model"""
        pass


    @property
    @abstractmethod
    def style_image(self):
        pass
    
    @style_image.setter
    @abstractmethod
    def style_image(self,image: torch.Tensor ):
        pass

    @property
    @abstractmethod
    def content_image(self):
        pass
    
    @content_image.setter
    @abstractmethod
    def content_image(self,image: torch.Tensor):
        pass
    
    

