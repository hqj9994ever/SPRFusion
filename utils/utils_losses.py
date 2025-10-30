import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from CLIP import clip
from model.SAM_encoder import get_encoder_base


from utils import utils_image as util


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
#load clip
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")
model.to(device)
for para in model.parameters():
    para.requires_grad = False



def get_clip_score(tensor, words):
    score=0
    text = clip.tokenize(words).to(device)
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img_resize = transforms.Resize((224, 224))
    for i in range(tensor.shape[0]):
        #image preprocess
        image2 = img_resize(tensor[i])
        image = clip_normalizer(image2).unsqueeze(0)
        #get probabilitis
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1)
        print(probs)
        # 2-word-compared probability
        # prob = probs[0][0]/probs[0][1] # you may need to change this line for more words comparison
        prob = probs[0][0]
        score = score + prob

    return score


class L_clip(nn.Module):
    def __init__(self):
        super(L_clip, self).__init__()
        for param in self.parameters(): 
            param.requires_grad = False
  
    def forward(self, x, light):
        k1 = get_clip_score(x, ["dark", "normal light"])
        if light:
            k2 = get_clip_score(x, ["noisy photo", "clear photo"])
            return (k1 + k2)/2
        return k1



def get_clip_score_from_feature(x, residual_vector1, residual_vector2, thr1=None, thr2=None):
    score = 0.0
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img_resize = transforms.Resize((224, 224))

    for i in range(x.shape[0]):
        image = img_resize(x[i])
        image = clip_normalizer(image.reshape(1, 3, 224, 224))
  
        image_features = model.encode_image(image)
        image_nor = image_features.norm(dim=-1, keepdim=True)
        nor1 = residual_vector1.norm(dim=-1, keepdim=True)
        nor2 = residual_vector2.norm(dim=-1, keepdim=True)

        if thr1 is None:
            similarity = (100.0 * (image_features/image_nor) @ (residual_vector1/nor1).T).softmax(dim=-1) + (100.0 * (image_features/image_nor) @ (residual_vector2/nor2).T).softmax(dim=-1)
        else:
            similarity = ((image_features/image_nor) @ (residual_vector1/nor1).T - thr1)**2 + ((image_features/image_nor) @ (residual_vector2/nor2).T - thr2)**2
        
        probs = similarity
        prob = probs[0][0]
        score = score + prob
    score = score / x.shape[0]
    return score


class L_clip_from_feature(nn.Module):
    def __init__(self):
        super(L_clip_from_feature, self).__init__()
        for param in self.parameters(): 
            param.requires_grad = False
  
    def forward(self, x, residual_vector1, residual_vector2, thr1, thr2):
        k = get_clip_score_from_feature(x, residual_vector1, residual_vector2, thr1, thr2)
        return k



# clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
# img_resize = transforms.Resize((224,224))

# def get_clip_score_from_feature(tensor, text_features):
# 	score=0
# 	for i in range(tensor.shape[0]):
# 		image2=img_resize(tensor[i])
# 		image=clip_normalizer(image2.reshape(1, 3, 224, 224))
  
# 		image_features = model.encode_image(image)
# 		image_nor=image_features.norm(dim=-1, keepdim=True)
# 		nor= text_features.norm(dim=-1, keepdim=True)
# 		similarity = (100.0 * (image_features/image_nor) @ (text_features/nor).T).softmax(dim=-1) # @是矩阵乘法操作
# 		probs = similarity
# 		# print(probs)
# 		prob = 1 - probs[0][-1]
# 		score = score + prob
# 	score = score / tensor.shape[0]
# 	return score


# class L_clip_from_feature(nn.Module):
# 	def __init__(self):
# 		super(L_clip_from_feature,self).__init__()
# 		for param in self.parameters(): 
# 			param.requires_grad = False
  
# 	def forward(self, x, text_features):
# 		k1 = get_clip_score_from_feature(x, text_features)
# 		return k1
    


class IlluminationLoss(nn.Module):
    def __init__(self, weight_A=0.5, weight_B=0.5):
        super(IlluminationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  
        self.weight_A = weight_A  
        self.weight_B = weight_B  

    def forward(self, f_A, f_B, f_gt):
        loss_A = self.mse_loss(f_A, f_gt)
        loss_B = self.mse_loss(f_B, f_gt)
        total_loss = self.weight_A * loss_A + self.weight_B * loss_B
        return total_loss

    

class FlowLoss(nn.Module):
    def __init__(self, gamma):
        super(FlowLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.gamma = gamma

    def forward(self, flow, flow_gt):
        n_predictions = len(flow)
        flow_loss = 0.0

        _, _, h, w = flow_gt.shape  # (n, 2, h, w)

        for i in range(n_predictions):
        
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = (flow[i] - flow_gt).abs()
            flow_loss += (i_weight * i_loss).mean()


        return flow_loss


class SAMFeatureExtractor(nn.Module):
    def __init__(self, sam_checkpoint="model_zoo/ckpt/sam_vit_b_01ec64.pth"):
        super().__init__()
        sam = get_encoder_base(checkpoint=sam_checkpoint)
        sam.eval() 
        for param in sam.parameters():
            param.requires_grad = False 

        self.image_encoder = sam.to(device)

        self.feat_layers = [2, 5, 8]  
        
        self.features = {}
        for layer_idx in self.feat_layers:
            self.image_encoder.blocks[layer_idx].register_forward_hook(
                lambda module, inp, out, idx=layer_idx: self._save_features(idx, out)
            )

    def _save_features(self, layer_idx: int, out: torch.Tensor):
        self.features[layer_idx] = out

    def forward(self, x: torch.Tensor):
        self.features.clear()
        _ = self.image_encoder(x)  
        return [self.features[idx] for idx in self.feat_layers] 


class MultiLayerSemanticLoss(nn.Module):
    def __init__(self):
        super(MultiLayerSemanticLoss, self).__init__()
        self.sam_extractor = SAMFeatureExtractor()
        self.instancenorm = nn.InstanceNorm2d(256, affine=False)
        self.feature_loss = nn.MSELoss()
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1))
    
    def sam_preprocess(self, x):
        # normalize to N(0, 1)
        x = (x * 255.0 - self.pixel_mean.to(x.device)) / self.pixel_std.to(x.device)
        return x

    def forward(self, x, target):
        x = self.sam_preprocess(x)
        target = self.sam_preprocess(target)

        x_feats = self.sam_extractor(x)
        target_feats = self.sam_extractor(target)
        b, c, h, w = x_feats[0].shape

        loss = 0.0
        
        for x_feat, target_feat in zip(x_feats, target_feats):
            loss += self.feature_loss(self.instancenorm(x_feat), self.instancenorm(target_feat))
            
        return loss


