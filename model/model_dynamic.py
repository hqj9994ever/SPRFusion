import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.nn.parallel import DataParallel

from collections import OrderedDict
import itertools
from tqdm import tqdm
from PIL import Image

from model.model_base import ModelBase
from model.network import define_A, define_F, define_D, define_G

from utils.utils_losses import L_clip_from_feature, IlluminationLoss, FlowLoss, MultiScaleSemanticLoss
from utils.utils import flow_warp2
from utils import utils_image as util

from CLIP import clip


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------
# load CLIP model
# ----------------------------------------
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")
model.to(device)
for para in model.parameters():
    para.requires_grad = False


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x


class Prompts(nn.Module):
    def __init__(self, initials=None, length_prompt=16):
        super(Prompts, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.length_prompt = length_prompt

        print("The initial prompts are:", initials)
        self.text_encoder = TextEncoder(model)
        if isinstance(initials, list):
            text = clip.tokenize(initials).to(self.device)
            self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_()).to(self.device)
        elif isinstance(initials, str):
            prompt_path = initials
            state_dict = torch.load(prompt_path)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.embedding_prompt = nn.Parameter(new_state_dict['embedding_prompt']).to(self.device)
            self.embedding_prompt.requires_grad = True
        else:
            self.embedding_prompt = torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding([" ".join(["X"]*self.length_prompt), " ".join(["X"]*self.length_prompt), " ".join(["X"]*self.length_prompt)]).requires_grad_())).to(self.device)

    def forward(self, tensor, flag=1):
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*self.length_prompt)]]) # opt.length_prompt
        text_features = self.text_encoder(self.embedding_prompt, tokenized_prompts) 
        for i in range(tensor.shape[0]):
            image_features = tensor[i]
            nor = torch.norm(text_features, dim=-1, keepdim=True)
            if flag == 0:
                similarity = (100.0 * image_features @ (text_features/nor).T)#.softmax(dim=-1)
                if(i == 0):
                    probs = similarity
                else:
                    probs = torch.cat([probs, similarity], dim=0)
            else:
                similarity = (100.0 * image_features @ (text_features/nor).T).softmax(dim=-1)#/nor
                if(i == 0):
                    probs = similarity[:,0]
                else:
                    probs = torch.cat([probs, similarity[:,0]], dim=0)
        return probs


class ModelPlain(ModelBase):
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        self.opt = opt
        self.stage = opt.train_stage
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.netA = define_A(in_channels=3, channels=64)
        self.netF = define_F()
        self.netD = define_D(in_channles=64)
        self.netG = define_G(in_channels=64)
        self.netA = self.model_to_device(self.netA)
        self.netF = self.model_to_device(self.netF)
        self.netD = self.model_to_device(self.netD)
        self.netG = self.model_to_device(self.netG)



    def init_train(self, init_paths):
        self.load(init_paths)
        # self.load_prompt()
        self.calculate_vectors()
        self.netA.train()
        self.netF.train()  
        self.netD.train()
        self.netG.train()
        self.define_loss()
        self.define_optimizer()
        self.define_scheduler()
        self.log_dict = OrderedDict()

    
    def model_to_device(self, net):
        net = net.to(self.device)
        net = DataParallel(net)
        return net


    def load(self, init_paths):
        load_path_A = init_paths[0]
        load_path_F = init_paths[1]
        load_path_D = init_paths[2]
        load_path_G = init_paths[3]

        if load_path_A is not None:
            print('Loading model for A [{:s}] ...'.format(load_path_A))
            self.load_network(load_path_A, self.netA, strict=True, param_key='params')
        
        if load_path_F is not None:
            print('Loading model for F [{:s}] ...'.format(load_path_F))
            self.load_network(load_path_F, self.netF, strict=True, param_key='params')

        if load_path_D is not None:
            print('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, strict=True, param_key='params')

        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=True, param_key='params')


    def load_prompt(self):
        # ----------------------------------------
        # load pretrained prompt
        # ----------------------------------------
        if self.opt.load_pretrain_prompt == True: 
            # l_learn_prompt=Prompts(self.opt.local_prompt_pretrain_dir, self.opt.length_prompt).to(self.device)
            g_learn_prompt=Prompts(self.opt.global_prompt_pretrain_dir, self.opt.length_prompt).to(self.device)
            # m_learn_prompt=Prompts(self.opt.mask_prompt_pretrain_dir, self.opt.length_prompt).to(self.device)
        else: 
            # l_learn_prompt=Prompts([" ".join(["X"]*(self.opt.length_prompt)), " ".join(["X"]*(self.opt.length_prompt)), " ".join(["X"]*(self.opt.length_prompt))], self.opt.length_prompt).to(self.device) 
            g_learn_prompt=Prompts([" ".join(["X"]*(self.opt.length_prompt)), " ".join(["X"]*(self.opt.length_prompt)), " ".join(["X"]*(self.opt.length_prompt))], self.opt.length_prompt).to(self.device) 
            # m_learn_prompt=Prompts([" ".join(["X"]*(self.opt.length_prompt)), " ".join(["X"]*(self.opt.length_prompt)), " ".join(["X"]*(self.opt.length_prompt))], self.opt.length_prompt).to(self.device) 

        # for k, v in l_learn_prompt.named_parameters():
        #     v.requires_grad_(False)
        for k, v in g_learn_prompt.named_parameters():
            v.requires_grad_(False)
        # for k, v in m_learn_prompt.named_parameters():
        #     v.requires_grad_(False)

        # l_learn_prompt = DataParallel(l_learn_prompt)
        g_learn_prompt = DataParallel(g_learn_prompt)
        # m_learn_prompt = DataParallel(m_learn_prompt)

        text_encoder = TextEncoder(model)
        text_encoder = DataParallel(text_encoder)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"] * self.opt.length_prompt)]])
        # l_embedding_prompt = l_learn_prompt.module.embedding_prompt 
        g_embedding_prompt = g_learn_prompt.module.embedding_prompt
        # m_embedding_prompt = m_learn_prompt.module.embedding_prompt
        # self.l_text_features = text_encoder(l_embedding_prompt, tokenized_prompts)
        self.g_text_features = text_encoder(g_embedding_prompt, tokenized_prompts)
        # self.m_text_features = text_encoder(m_embedding_prompt, tokenized_prompts)


    def save_vectors(self, save_dir="model_zoo"):
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'residual_vector1': self.residual_vector1,
            'residual_vector2': self.residual_vector2,
            'thr1': self.thr1,
            'thr2': self.thr2,
        }, os.path.join(save_dir, 'light_vectors.pt'))
        
        print(f"Vectors have been saved in {save_dir}/light_vectors.pt")


    def load_vectors(self, save_dir="model_zoo"):
        path = os.path.join(save_dir, 'light_vectors.pt')
        if os.path.exists(path):
            data = torch.load(path)
            self.residual_vector1 = data['residual_vector1'].to(self.device)
            self.residual_vector2 = data['residual_vector2'].to(self.device)
            self.thr1 = data['thr1'].to(self.device)
            self.thr2 = data['thr2'].to(self.device)
            print("Vector loading successful.")
            return True
        else:
            print(f"The saved vector file {path} was not found")
            return False


    def calculate_vectors(self):

        if self.load_vectors():
            return
        
        embs_neg1 = self.extract_embs(self.config.lowlight_images_path, model)
        embs_neg2 = self.extract_embs(self.config.overlight_images_path, model)
        embs_pos = self.extract_embs(self.config.normallight_images_path, model)

        vector_pos = torch.mean(embs_pos, dim=0)
        vector_pos = vector_pos / torch.norm(vector_pos)

        vector_neg1 = torch.mean(embs_neg1, dim=0)
        vector_neg1 = vector_neg1 / torch.norm(vector_neg1)

        vector_neg2 = torch.mean(embs_neg2, dim=0)
        vector_neg2 = vector_neg2 / torch.norm(vector_neg2)

        residual_vector1 = vector_pos - vector_neg1
        residual_vector1 = residual_vector1 / torch.norm(residual_vector1)

        residual_vector2 = vector_pos - vector_neg2
        residual_vector2 = residual_vector2 / torch.norm(residual_vector2)

        self.thr1 = torch.dot(vector_pos, residual_vector1)
        self.residual_vector1 = residual_vector1.view(1, 512).to(self.device)

        self.thr2 = torch.dot(vector_pos, residual_vector2)
        self.residual_vector2 = residual_vector2.view(1, 512).to(self.device)
        
        self.save_vectors()


    def define_loss(self):
        self.L_clip = L_clip_from_feature().to(self.device)
        self.L_sam = MultiScaleSemanticLoss().to(self.device)
        self.L_illumination = IlluminationLoss(weight_A=1.0, weight_B=1.0).to(self.device)
        self.L_flow = FlowLoss(gamma=0.9).to(self.device)
        self.L_content = nn.L1Loss().to(self.device)


    def define_optimizer(self):
        A_optim_params = []
        for k, v in self.netA.named_parameters():
            if v.requires_grad:
                A_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        F_optim_params = []
        for k, v in self.netF.named_parameters():
            if v.requires_grad:
                F_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k)) 

        D_optim_params = []
        for k, v in self.netD.named_parameters():
            if v.requires_grad:
                D_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k)) 
        
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        self.train_optimizerF = torch.optim.Adam(itertools.chain(A_optim_params, F_optim_params), lr=self.opt.train_lr, betas=(0.5, 0.999))
        self.train_optimizerG = torch.optim.Adam(itertools.chain(D_optim_params, G_optim_params), lr=self.opt.train_lr, betas=(0.5, 0.999))
    

    def define_scheduler(self):
        self.lr_schedulerF = lr_scheduler.MultiStepLR(self.train_optimizerF, milestones=[self.opt.num_epochs/4, self.opt.num_epochs/2, self.opt.num_epochs/4*3], gamma=0.4)
        self.lr_schedulerG = lr_scheduler.MultiStepLR(self.train_optimizerG, milestones=[self.opt.num_epochs/4, self.opt.num_epochs/2, self.opt.num_epochs/4*3], gamma=0.4)    


    def feed_data(self, under, over, gt, flow_gt=None, under_g=None, over_g=None, image_path=None):
        self.under = under.to(self.device)
        self.over = over.to(self.device)
        self.under_g = under_g.to(self.device) if under_g is not None else None
        self.over_g = over_g.to(self.device) if over_g is not None else None
        self.gt = gt.to(self.device) if gt is not None else None
        self.flow_gt = flow_gt.to(self.device) if flow_gt is not None else None
        self.image_path = image_path if image_path is not None else None
        if self.flow_gt is not None:
            self.over_warp = flow_warp2(self.over, self.flow_gt.permute(0, 2, 3, 1))
            self.over_g_warp = flow_warp2(self.over_g, self.flow_gt.permute(0, 2, 3, 1))
        else:
            self.over_warp = self.over

        
    def netA_forward(self):
        self.identity_u, self.correct_u = self.netA(self.under)
        self.identity_o, self.correct_o = self.netA(self.over_warp)

        if self.under_g is not None:
            self.identity_u_g, self.correct_u_g = self.netA(self.under_g)
            self.identity_o_g, self.correct_o_g = self.netA(self.over_g)


    def netF_forward(self):
        self.align_o, self.f_w, self.f_s = self.netF(self.correct_o, self.correct_u, self.identity_o)
        if self.under_g is not None:
            self.align_o_g, _, _ = self.netF(self.correct_o_g, self.correct_u_g, self.identity_o_g)


    def netD_forward(self):
        self.align_u, self.align_o = self.netD(self.identity_u, self.align_o)
        if self.under_g is not None:
            self.align_u_g, self.align_o_g = self.netD(self.identity_u_g, self.align_o_g)
    
    
    def netG_forward(self):
        self.output = self.netG(self.align_u, self.align_o) 
        if self.under_g is not None:
            self.output_g = self.netG(self.align_u_g, self.align_o_g)


    def optimize_parametersF(self):
        self.train_optimizerF.zero_grad()
        self.netA_forward()
        self.netF_forward()
        loss_illumination = self.L_illumination(self.correct_u, self.correct_o, self.gt)
        loss_flow = self.L_flow(self.f_s, self.flow_gt)
        loss_reg = loss_illumination + loss_flow
        loss_reg.backward()
        self.train_optimizerF.step()
        self.log_dict['loss_flow'] = loss_flow.item()
        self.log_dict['loss_illumination'] = loss_illumination.item()


    def optimize_parametersG(self):
        self.train_optimizerG.zero_grad()
        self.netD_forward()
        self.netG_forward()
        loss_content = self.L_content(self.output, self.gt)
        # loss_sam = self.L_sam(self.output, self.gt)
        # loss_clip_local = self.L_clip(self.output, self.residual_vector1, self.residual_vector2, self.thr1, self.thr2)
        # loss_clip_global = self.L_clip(self.output_g, self.residual_vector1, self.residual_vector2, self.thr1, self.thr2)
       
        # loss_fusion = loss_content + 1e-3 * loss_clip_local + 1e-3 * loss_clip_global + 1e-2 * loss_sam
        loss_fusion = loss_content 

        loss_fusion.backward()
        self.train_optimizerG.step()
        self.log_dict['loss_content'] = loss_content.item()
        # self.log_dict['loss_clip_local'] = loss_clip_local.item()
        # self.log_dict['loss_clip_global'] = loss_clip_global.item()
        # self.log_dict['loss_sam'] = loss_sam.item()
        
        
    def optimize_parametersGF(self):
        self.train_optimizerF.zero_grad()
        self.netA_forward()
        self.netF_forward()
        with torch.no_grad():
            self.netD_forward()
            self.netG_forward()
        loss_illumination = self.L_illumination(self.correct_u, self.correct_o, self.gt)
        loss_flow = self.L_flow(self.f_s, self.flow_gt)
        loss_reg = loss_illumination + loss_flow
        loss_content = self.L_content(self.output, self.gt)
        loss_sam = self.L_sam(self.output, self.gt)
        loss_clip_local = self.L_clip(self.output, self.residual_vector1, self.residual_vector2, self.thr1, self.thr2)
        loss_clip_global = self.L_clip(self.output_g, self.residual_vector1, self.residual_vector2, self.thr1, self.thr2)
        # loss_clip = self.L_clip(self.output, self.g_text_features)
       
        loss_fusion = loss_content + 1e-2 * loss_sam  + 1e-3 * loss_clip_global + 1e-3 * loss_clip_local 

        total_loss = 1e-2 * loss_reg + loss_fusion 
        total_loss.backward()
        self.train_optimizerF.step()

        self.train_optimizerG.zero_grad()
        with torch.no_grad():
            self.netA_forward()
            self.netF_forward()
        self.netD_forward()
        self.netG_forward()
        loss_content = self.L_content(self.output, self.gt)
        loss_sam = self.L_sam(self.output, self.gt)
        loss_clip_local = self.L_clip(self.output, self.residual_vector1, self.residual_vector2, self.thr1, self.thr2)
        loss_clip_global = self.L_clip(self.output_g, self.residual_vector1, self.residual_vector2, self.thr1, self.thr2)
        # loss_clip = self.L_clip(self.output, self.g_text_features)
       
        loss_fusion = loss_content + 1e-2 * loss_sam + 1e-3 * loss_clip_global + 1e-3 * loss_clip_local  
        loss_fusion.backward()
        self.train_optimizerG.step()
        self.log_dict['loss_flow'] = loss_flow.item()
        self.log_dict['loss_illumination'] = loss_illumination.item()
        self.log_dict['loss_content'] = loss_content.item()
        self.log_dict['loss_clip_local'] = loss_clip_local.item()
        self.log_dict['loss_clip_global'] = loss_clip_global.item()
        # self.log_dict['loss_clip'] = loss_clip.item()
        self.log_dict['loss_sam'] = loss_sam.item()
        


    def optimize_parameters(self):
        if self.stage == 'align':
            self.optimize_parametersF()
        elif self.stage == 'fusion':
            with torch.no_grad():
                self.netA_forward()
                self.netF_forward()
            self.optimize_parametersG()
        elif self.stage == 'joint':
            self.optimize_parametersGF()

    def update_learning_rate(self, epoch):
        if self.stage == 'align':
            self.lr_schedulerF.step(epoch)
        elif self.stage == 'fusion':
            self.lr_schedulerG.step(epoch)
        elif self.stage == 'joint':
            self.lr_schedulerF.step(epoch)
            self.lr_schedulerG.step(epoch)


    def test(self):
        self.netA.eval()
        self.netF.eval()
        self.netD.eval()
        self.netG.eval()
        with torch.no_grad():
            self.netA_forward()
            self.netF_forward()
            self.netD_forward()
            self.netG_forward()
        self.netA.train()
        self.netF.train()
        self.netD.train()
        self.netG.train()
    

    def current_log(self):
        return self.log_dict
    

    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['E'] = self.output.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.gt.detach()[0].float().cpu()
        return out_dict


    def print_network(self):
        num_params = 0
        for param in self.netA.parameters():
            num_params += param.numel()
        for param in self.netF.parameters():
            num_params += param.numel()
        for param in self.netD.parameters():
            num_params += param.numel()
        for param in self.netG.parameters():
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)


    def save(self, epoch):
        if self.stage == 'align':
            self.save_network(self.save_dir, self.netA, 'A', epoch)
            self.save_network(self.save_dir, self.netF, 'F', epoch)
        elif self.stage == 'fusion':
            self.save_network(self.save_dir, self.netD, 'D', epoch)
            self.save_network(self.save_dir, self.netG, 'G', epoch)
        elif self.stage == 'joint':
            self.save_network(self.save_dir, self.netA, 'A', epoch)
            self.save_network(self.save_dir, self.netF, 'F', epoch)
            self.save_network(self.save_dir, self.netD, 'D', epoch)
            self.save_network(self.save_dir, self.netG, 'G', epoch)

    
    def current_learning_rate(self):
        if self.stage == 'align':
            return self.lr_schedulerF.get_lr()[0]
        elif self.stage == 'fusion':
            return self.lr_schedulerG.get_lr()[0]
        elif self.stage == 'joint':
            return self.lr_schedulerF.get_lr()[0]


    def extract_embs(self, data_path, model):
        model_embs = None
        
        for img_name in tqdm(os.listdir(data_path)):
            try:
                img = Image.open(os.path.join(data_path, img_name))
            except Exception as e:
                continue

            # Preprocess and move to device
            inputs = preprocess(img).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
            
            # Extract features
            with torch.no_grad():  
                outputs = model.visual(inputs)
            
            # L2 normalize 
            cls = outputs / outputs.norm(dim=1, keepdim=True) 
            
            # Concatenate embeddings
            if model_embs is None:
                model_embs = cls
            else:
                model_embs = torch.cat((model_embs, cls), dim=0)  
        
        return model_embs

