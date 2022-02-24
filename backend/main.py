from nst import NST
from torch.optim import Adam
from cnn_models import Vgg19Model

if __name__ == '__main__':

    style_transferer = NST(
        content_image_path= './content.jpg',
        style_image_path='./style.jpg',
        content_layer_index=22,
        style_layer_indices=[(0,0.75),(5,0.5),(10,0.5),(19,0.3),(25,0.3),(34,0.3)],
        content_weight=1e4,
        style_weight=1e2,
        epoch=1000,
        cnn_model = Vgg19Model
    )
    style_transferer.train(optimizer=Adam)

