import enum
import torch
import torch.nn as nn
import torchvision.transforms as tf
from torchvision.models import vgg19
from interfaces import CNNModel, Calculatable, FeatureExtractor
from typing import List, Tuple, Union

class ImageType(enum.Enum):

    style = 1
    content = 2
    target = 3

class Vgg19FeatureExtractor(FeatureExtractor):

    def __init__(self,img_tensor: torch.Tensor, pretrained_model: torch.nn.Module) -> None:
        """_summary_

        Args:
            img_tensor (torch.Tensor): _description_
            pretrained_model (torch.nn.Module): _description_
        """
        super(Vgg19FeatureExtractor,self).__init__(img_tensor=img_tensor,pretrained_model=pretrained_model)


        self.__image = img_tensor
        self.model = pretrained_model
    
    def __get_feature_list(self,layer_indices: List[int]):
        """get list of feature extracted from the model 

        Args:
            layer_indices (List[int]): list of layer index
            from which the feature will be extracted from

        Returns:
            List[torch.Tensor]: list of extracted features
        """
        features = list()

        x = self.__image

        for i,layer in enumerate(self.model.features):

            x = layer(x)
            if i in layer_indices:
                
                features.append(x)
        
        return features

    def __get_single_feature(self,layer_index:int):
        """get a single feature extracted from the model

        Args:
            layer_index (int): index of the cnn layer
            from which the feature will be extracted

        Returns:
            torch.Tensor : extracted feature
        """
        feature = None

        x = self.__image
        
        for i, layer in enumerate(self.model.features):

            if i == layer_index:
                
                feature = layer(x)
                break
            else:
                x = layer(x)

        return feature

    def get_features(self,layer_indices:Union[List[int],int]):

        if isinstance(layer_indices,int):
            return self.__get_single_feature(layer_index=layer_indices)
        
        if isinstance(layer_indices,list):
            return self.__get_feature_list(layer_indices=layer_indices)




class Vgg19Model(CNNModel):

    def __init__(
        self,content_image: torch.Tensor, 
        style_image: torch.Tensor,
        content_layer_index: int,
        style_layer_indices: List[Tuple[int,float]]
    ):
        """subclass of CNNModel that implements neural style transfer model
        based on vgg19 (pretrained)

        Args:
            content_image (torch.Tensor): content image on which the style transfer
            will be performed
            style_image (torch.Tensor): image from which the style feature will be taken
            content_layer_index (int): index of CNN layer from which the feature
            will be extracted using content image
            style_layer_indices (List[Tuple[int,float]]): a list of tuple containing 
            layer indices and the corresponding weights to be imposed for generating style
        """

        super(Vgg19Model,self).__init__()

        self.__content_image = content_image
        self.__style_image = style_image
        self.__content_layer_index = content_layer_index
        self.__style_layer_indices = [s for s,w in style_layer_indices]
        self.__style_layer_weight = [w for s,w in style_layer_indices]
        self.__pretrained_model = vgg19(pretrained=True)
        self.model = self.__pretrained_model

    @property
    def style_image(self):
        return self.__style_image

    @property
    def content_image(self):
        return self.__content_image

    @property
    def content_layer_index(self):
        return self.__content_layer_index

    @property
    def style_layer_indices(self):
        return self.__style_layer_indices

    @property
    def style_layer_weight(self):
        return self.__style_layer_weight

    @property
    def style_features(self):

        feature_extractor = Vgg19FeatureExtractor(
            img_tensor=self.style_image,
            pretrained_model=self.model
        )
        style_features = feature_extractor.get_features(self.style_layer_indices)
        
        return style_features

    @property
    def content_features(self):

        feature_extractor = Vgg19FeatureExtractor(
            img_tensor=self.content_image,
            pretrained_model=self.model
        )
        content_feature = feature_extractor.get_features(layer_indices=self.content_layer_index)
        
        return content_feature
    
    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self,pretrained_model=vgg19):

        for params in pretrained_model.parameters():
            params.requires_grad = False
        
        pretrained_model.cuda().eval()

        for i, layer in enumerate(pretrained_model.features):
            
            if isinstance(layer, torch.nn.MaxPool2d):
                pretrained_model.features[i] = torch.nn.AvgPool2d(
                    kernel_size=2,
                    stride=2,
                    padding=0
                )
        
        self.__model = pretrained_model

    def get_target_features(self, target_img: torch.Tensor):

        feature_extractor = Vgg19FeatureExtractor(img_tensor=target_img,pretrained_model=self.model)
        target_feature_style = feature_extractor.get_features(layer_indices=self.style_layer_indices)
        target_feature_content = feature_extractor.get_features(layer_indices=self.content_layer_index)


        return {
            "style" : target_feature_style,
            "content" : target_feature_content
        }
    
    
    

        







        