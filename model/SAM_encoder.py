import torch
from .SAM.image_encoder import ImageEncoderViT
from functools import partial


def get_encoder(checkpoint = None, ft_ckpt = False):
    encoder = ImageEncoderViT(depth=32,
            embed_dim=1280,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=16,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[7, 15, 23, 31],
            window_size=14,
            out_chans=256,
            pos_crop_v0 = ft_ckpt)
    
    encoder.eval()
    
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')
        new_state_dict = encoder.state_dict()
        for k in new_state_dict.keys():
            full_k = 'image_encoder.' + k
            new_state_dict[k] = state_dict[full_k]
        encoder.load_state_dict(new_state_dict)

    return encoder

def get_encoder_base(checkpoint = None):
    encoder = ImageEncoderViT(depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,)
    
    encoder.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')
        new_state_dict = encoder.state_dict()
        for k in new_state_dict.keys():
            full_k = 'image_encoder.' + k
            new_state_dict[k] = state_dict[full_k]
        encoder.load_state_dict(new_state_dict)

    return encoder


if __name__ == '__main__':
    from thop import profile
    encoder = get_encoder_base()
    inputs = torch.randn(1, 3, 256, 256)
    # flops, params = profile(encoder, (inputs,))
    # print('flops: ', flops, 'params: ', params)
    outputs = encoder(inputs)
    print(outputs.shape)