from models.mobilenet import MyMobilenet
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

CONFIG_METABLOCK_BY_MODEL = {
    'mobilenet-v3-small': 40,
}

def set_class_model (model_name, num_class, comb_method=None, comb_config=None):

    model = None
    if model_name == 'mobilenet-v3-small':
        model = MyMobilenet(models.mobilenet_v3_large(weights=MobileNet_V3_Small_Weights), num_class, comb_method=comb_method, comb_config=comb_config)
    else:
        raise ValueError(f"The model {model_name} is not available!")
                             
    return model

