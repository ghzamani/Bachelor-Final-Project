from transformers import AutoModel, CLIPVisionModel, RobertaModel, AutoTokenizer, CLIPFeatureExtractor
from torch import nn
from transformers import CLIPConfig, CLIPModel
# from ..config import TEXT_MODEL, IMAGE_MODEL

# def clip_wraper_creator():
#     """create a dummy CLIPModel to wrap text and vision encoders in order to use CLIPTrainer"""
#     config = {'num_hidden_layers': 0,
#               'max_position_embeddings': 0,
#               'vocab_size': 0,
#               'hidden_size': 1,
#               'patch_size': 1,
#               }
#     DUMMY_CONFIG = CLIPConfig(text_config_dict=config,
#                               vision_config_dict=config)
#     clip = CLIPModel(config=DUMMY_CONFIG)
#     # convert projectors to Identity
#     clip.text_projection = nn.Identity()
#     clip.visual_projection = nn.Identity()
#     return clip

def get_model(model_weights):
    clip = CLIPModel.from_pretrained(model_weights)
    # vision_encoder = CLIPVisionModel.from_pretrained(IMAGE_MODEL)
    # text_encoder = AutoModel.from_pretrained(TEXT_MODEL)
    # vision_encoder = CLIPVisionModel.from_pretrained('SajjadAyoubi/clip-fa-vision')
    # # preprocessor = CLIPFeatureExtractor.from_pretrained('SajjadAyoubi/clip-fa-vision')
    # text_encoder = RobertaModel.from_pretrained('SajjadAyoubi/clip-fa-text')
    # # tokenizer = AutoTokenizer.from_pretrained('SajjadAyoubi/clip-fa-text')
    # assert text_encoder.config.hidden_size == vision_encoder.config.hidden_size

    # clip = clip_wraper_creator()
    # clip.text_model = text_encoder
    # clip.vision_model = vision_encoder
    return clip