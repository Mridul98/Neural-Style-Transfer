import argparse
from ast import arg
import json
from nst import NST
from torch.optim import Adam
from cnn_models import Vgg19Model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nst_params_json_string', type=str, required=True, help='define parameters for Neural Style Transfer module')
    
    arguments = {
        "content_image_path" : "./content.jpg",
        "style_image_path" : "./style.jpg",
        "content_layer_index" : 22,
        "style_layer_index" : [(0,0.75),(5,0.5),(10,0.5),(19,0.3),(25,0.3),(34,0.3)],
        "content_weight": 1e4,
        "style_weight" : 1e2,
        "epoch" : 1000
    }
    
    style_transferer = NST(
        content_image_path=arguments['content_image_path'],
        style_image_path=arguments['style_image_path'],
        content_layer_index=arguments['content_layer_index'],
        style_layer_indices=arguments['style_layer_index'],
        content_weight=float(arguments['content_weight']),
        style_weight=float(arguments['style_weight']),
        epoch=int(arguments['epoch']),
        cnn_model = Vgg19Model
    )
    style_transferer.train(optimizer=Adam)

    # demo_json = {
    #         "content_image_path" : "./content.jpg",
    #         "style_image_path" : "./style.jpg",
    #         "content_layer_index" : 22,
    #         "style_layer_index" : [(0,0.75),(5,0.5),(10,0.5),(19,0.3),(25,0.3),(34,0.3)],
    #         "content_weight": 1e4,
    #         "style_weight" : 1e2,
    #         "epoch" : 1000
    # }

    # print(json.dumps(demo_json))
    

    