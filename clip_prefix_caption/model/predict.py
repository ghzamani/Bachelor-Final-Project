# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    CLIPFeatureExtractor,
    CLIPVisionModel
)
import skimage.io as io
import PIL.Image
import argparse
from enum import Enum
import json

# import cog

# import torch

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

WEIGHTS_PATHS = {
    "coco": "coco_weights.pt",
    "conceptual-captions": "conceptual_weights.pt",
}

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

D = torch.device
CPU = torch.device("cpu")

class Predictor:
    def __init__(self, model_weights_path, mapping_type, clip_length, num_layers, is_eng=True, training_model=None, prefix_length=40, prefix_size=640, clip_model="RN50x4"):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.device = torch.device("cuda")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("the device is", self.device)
        self.is_eng = is_eng
        if is_eng:
            self.clip_model, self.preprocess = clip.load(
                clip_model, device=self.device, jit=False
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            self.clip_model = CLIPVisionModel.from_pretrained('SajjadAyoubi/clip-fa-vision')
            self.preprocess = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian')

        self.prefix_length = prefix_length

        if not training_model:
            model = ClipCaptionModel(self.prefix_length,mapping_type,clip_length, num_layers, is_eng, prefix_size)
            model.load_state_dict(torch.load(model_weights_path, map_location=CPU))
            model = model.eval()
            model = model.to(self.device)
            self.model = model
        else:
            self.model = training_model

    def predict(self, image, use_beam_search=False):
        """Run a single prediction on the model"""
        image = io.imread(image)
        model = self.model
        if self.is_eng:
            pil_image = PIL.Image.fromarray(image)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prefix = self.clip_model.encode_image(image).to(
                    self.device, dtype=torch.float32
                )
                prefix_embed = model.clip_project(prefix).reshape(1, self.prefix_length, -1)
        else:
            inputs = self.preprocess(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                prefix = outputs.last_hidden_state.to(
                    self.device, dtype=torch.float32
                )
                prefix_embed = model.clip_project(prefix).reshape(1, self.prefix_length, -1)
        if use_beam_search:
            return generate_beam(model, self.tokenizer, embed=prefix_embed)[0]
        else:
            return generate2(model, self.tokenizer, embed=prefix_embed)
        
    def test(self, test_path, use_beam_search=False):
        with open(test_path, 'r') as f:
            test_dataset = json.load(f)
        results = []
        for data in test_dataset:
            results.append(self.predict(data['image'], use_beam_search=use_beam_search))
        return results

# class Predictor(cog.Predictor):
#     def setup(self):
#         """Load the model into memory to make running multiple predictions efficient"""
#         self.device = torch.device("cuda")
#         self.clip_model, self.preprocess = clip.load(
#             "ViT-B/32", device=self.device, jit=False
#         )
#         self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#         self.models = {}
#         self.prefix_length = 10
#         for key, weights_path in WEIGHTS_PATHS.items():
#             model = ClipCaptionModel(self.prefix_length)
#             model.load_state_dict(torch.load(weights_path, map_location=CPU))
#             model = model.eval()
#             model = model.to(self.device)
#             self.models[key] = model

#     @cog.input("image", type=cog.Path, help="Input image")
#     @cog.input(
#         "model",
#         type=str,
#         options=WEIGHTS_PATHS.keys(),
#         default="coco",
#         help="Model to use",
#     )
#     @cog.input(
#         "use_beam_search",
#         type=bool,
#         default=False,
#         help="Whether to apply beam search to generate the output text",
#     )
#     def predict(self, image, model, use_beam_search):
#         """Run a single prediction on the model"""
#         image = io.imread(image)
#         model = self.models[model]
#         pil_image = PIL.Image.fromarray(image)
#         image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             prefix = self.clip_model.encode_image(image).to(
#                 self.device, dtype=torch.float32
#             )
#             prefix_embed = model.clip_project(prefix).reshape(1, self.prefix_length, -1)
#         if use_beam_search:
#             return generate_beam(model, self.tokenizer, embed=prefix_embed)[0]
#         else:
#             return generate2(model, self.tokenizer, embed=prefix_embed)


class MLP(nn.Module):
    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, mapping_type, clip_length, num_layers, is_eng, prefix_size: int = 640):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        if is_eng:
            self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        else:
            self.gpt = GPT2LMHeadModel.from_pretrained('bolbolzaban/gpt2-persian')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        # if prefix_length > 10:  # not enough memory
        #     self.clip_project = nn.Linear(
        #         prefix_size, self.gpt_embedding_size * prefix_length
        #     )
        # else:
            # if mapping_type == MappingType.MLP:
            #     self.clip_project = MLP(
            #         (
            #             prefix_size,
            #             (self.gpt_embedding_size * prefix_length) // 2,
            #             self.gpt_embedding_size * prefix_length,
            #         )
            #     )
            # else:
            #     self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                    #  clip_length, num_layers)
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)




def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break
            print(tokens)
            print(tokens.squeeze())
            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights_path')
    parser.add_argument('--image_path')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--prefix_length', type=int, default=40)
    parser.add_argument('--prefix_size', type=int, default=640)
    parser.add_argument('--language', default="english", choices=('english', 'persian'))
    parser.add_argument('--clip_model_type', default="RN50x4", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    predictor = Predictor(args.model_weights_path, mapping_type=args.mapping_type,clip_length=args.prefix_length_clip, num_layers=args.num_layers, is_eng= (args.language == "english"),prefix_length=args.prefix_length, prefix_size=args.prefix_size, clip_model=args.clip_model_type)
    print(predictor.predict(args.image_path, use_beam_search=True))
    # exit(main(args.clip_model_type))
