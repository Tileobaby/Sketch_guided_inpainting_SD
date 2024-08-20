"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

from LGP import save_out_hook,resize_and_concatenate###

from PIL import Image


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    #@torch.no_grad()
    def sample(self,
               model_LGP,
               sketch,
               middle_number,
               image_m,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               sketch_scale = 0.05,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(model_LGP,sketch,middle_number,image_m,conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    sketch_scale = sketch_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    #@torch.no_grad()
    def ddim_sampling(self, model_LGP, sketch, middle_number,image_m, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1.,sketch_scale=0.05, unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        
        #print(iterator)###
        #edge_map = Image.open('/home/tianle/Desktop/stable-diffusion/edge_horse.png')#read the reference image
        #edge_map = edge_map.resize((512,512))
        #edge_map = (np.array(edge_map).astype(np.float32) / 255.0) * 2.0 - 1.0
        edge_map = (sketch/255)*2-1
        #print(edge_map.shape)#
        #edge_map = edge_map[None].transpose(0, 3, 1, 2)
        #edge_map = torch.from_numpy(edge_map)
        edge_map = edge_map.to(device=device)
        encoded_edge = self.model.get_first_stage_encoding(self.model.encode_first_stage(edge_map))
        #print(encoded_edge.shape)###
        
        #print(dir(self.model.model))


        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            ##print(ts,index,i,step)##tensor([321], device='cuda:0') 16 33 321

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img #combine the noised masked image with generated image 
            
            #print(ts,index,ddim_use_original_steps)###output: [1,1000],[1,50],False

            outs = self.p_sample_ddim(model_LGP, img, encoded_edge, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      sketch_scale = sketch_scale,
                                      unconditional_conditioning=unconditional_conditioning)# each iteration
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            #add the original one:
            if ts >= middle_number:
                mean = (extract_into_tensor(self.sqrt_alphas_cumprod, ts, image_m.shape) * image_m)
                variance = extract_into_tensor(1.0 - self.alphas_cumprod, ts, image_m.shape)
                img = torch.normal(mean, torch.sqrt(variance))
                #ts is smaller and smaller

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        #print(img.shape)#1,4,64,64
        #print(mask)#None

        return img, intermediates

    #@torch.no_grad()
    def p_sample_ddim(self, model_LGP, x, encoded_edge, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., sketch_scale=0.05,  unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        ###
        x.detach_()
        x.requires_grad = True
        x.retain_grad()

        blocks = [0,1,3]

        #print(dir(self.model.model.diffusion_model))##
        #print(dir(UNetModel))#


        activations = []
        h_m = []
        h_in = []
        h_out = []
        save_hook = save_out_hook
        feature_blocks = []
        for idx, block in enumerate(self.model.model.diffusion_model.input_blocks):
            if idx in blocks:
                h = block.register_forward_hook(save_hook)
                h_in.append(h)
                feature_blocks.append(block) 
                
        for idx, block in enumerate(self.model.model.diffusion_model.output_blocks):
            if idx in blocks:
                h = block.register_forward_hook(save_hook)
                h_out.append(h)
                feature_blocks.append(block)

        block = self.model.model.diffusion_model.middle_block
        h_m = block.register_forward_hook(save_hook)
        feature_blocks.append(block)
        ###

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [
                            torch.cat([unconditional_conditioning[k][i], c[k][i]])
                            for i in range(len(c[k]))
                        ]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            
            ###print(type(c_in),len(c_in),c_in['c_concat'][0].shape,c_in['c_crossattn'][0].shape)
            ###<class 'dict'> 2 torch.Size([2, 5, 64, 64]) torch.Size([2, 77, 768])

            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)### core

            #print(x_in.shape,t_in.shape)#???

            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)#add the scale of conditional guidence

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas# get the noise
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature##noise
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        

        # Extract activations
        for block in feature_blocks:
            activations.append(block.activations)
            block.activations = None
            
        activations = [activations[0], activations[1], activations[2], activations[3], activations[4], activations[5], activations[6]]
        
        #print(activations[0].shape,activations[1].shape,activations[2].shape,activations[3].shape,activations[4].shape,activations[5].shape,activations[6].shape)#
        #output:torch.Size([2, 320, 56, 56]) torch.Size([2, 320, 56, 56]) torch.Size([2, 320, 28, 28]) 
        # torch.Size([2, 1280, 7, 7]) torch.Size([2, 1280, 7, 7]) torch.Size([2, 1280, 14, 14]) torch.Size([2, 1280, 7, 7])


        for i in range(3):#remove the hook
            h_in[i].remove()
            h_out[i].remove()
        h_m.remove()

        features = resize_and_concatenate(activations, x)#6080

        #features.requires_grad = True###

        #noise_level = sqrt_one_minus_alphas[index] * torch.ones(x.shape, device=device)##
        noise_level = noise.transpose(1,3)        

        pred_edge_map = model_LGP(features, noise_level).unflatten(0, (1, 64, 64)).transpose(3, 1)#6120
        
        #try to replace the noise_level with noise
        #print(type(noise),noise.shape,noise.mean())#<class 'torch.Tensor'> torch.Size([1, 4, 64, 64]) tensor(-7.7611e-05, device='cuda:0')
        #print(type(noise_level),noise_level.shape,noise.mean())#<class 'torch.Tensor'> torch.Size([1, 64, 64, 4]) tensor(-7.7611e-05, device='cuda:0')


        #criterion = torch.nn.MSELoss()##
        #loss = criterion(pred_edge_map, encoded_edge)##
        #print(loss)##
        """
        pred_edge_map = self.model.decode_first_stage(pred_edge_map)
        #print(pred_edge_map.shape)##
        pred_edge_map = torch.clamp((pred_edge_map+1.0)/2.0, min=0.0, max=1.0)
        pred_edge_map = pred_edge_map.cpu().numpy().transpose(0,2,3,1)
        pred_edge_map = pred_edge_map *255
        #print(pred_edge_map.shape)##

        pred_edge_map = pred_edge_map.squeeze(0)
        pred_edge_map = Image.fromarray(pred_edge_map.astype(np.uint8))
        pred_edge_map.save(str(index)+'pred_edge.jpg')
        """

        #print(f"{'befor loss'}: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")####
        print(f"{'befor loss'}: {torch.cuda.memory_reserved() / 1024**2:.2f} MB allocated")####
        if t > 200:
            loss_function = torch.nn.MSELoss()
            loss = torch.log(loss_function(pred_edge_map, encoded_edge))
            loss.backward(inputs = x)
            gradient_noisy_image = x.grad
            
            #print("loss:", loss)
            #print("x.is_leaf: ", x.is_leaf)###
            #print("x.grad after backward:", gradient_noisy_image)

            alpha = sketch_scale * torch.norm(x_prev - x ,p=2) / torch.norm(gradient_noisy_image, p=2)

            x_prev = x_prev - alpha * gradient_noisy_image
        
        return x_prev, pred_x0

    #@torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    #@torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
