import sys
import numpy as np
import streamlit as st
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from main import instantiate_from_config
from streamlit_drawable_canvas import st_canvas
import torch

import os


from ldm.models.diffusion.ddim import DDIMSampler

from LGP import latent_guidance_predictor,extract_features, resize_and_concatenate


MAX_SIZE = 640

# load safety model
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from imwatermark import WatermarkEncoder
import cv2

#torch.cuda.memory._record_memory_history()###

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
wm = "StableDiffusionV1-Inpainting"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def put_watermark(img):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    return x_checked_image, has_nsfw_concept


@st.cache(allow_output_mutation=True)
#@st.cache_resource
def initialize_model(config, ckpt):

    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    

    model.load_state_dict(torch.load(ckpt,map_location="cpu")["state_dict"], strict=False)
    
    model.eval()###

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    #buffer ft16
    #for name, buffer in model.named_buffers():
    #    buffer_fp16 = buffer.half().cpu()
    #    setattr(model, name, buffer_fp16)
    

    sampler = DDIMSampler(model)

    total_params = sum(p.numel() for p in model.parameters())###
    total_buffers = sum(p.numel() for p in model.buffers())
    print(f"{model.__class__.__name__} has {(total_params+total_buffers) * 1.e-6:.2f} M params.")###
    
    #check the type and size of the parameters
    #param_types = [[param.dtype,param.size()] for param in model.parameters()]##
    #with open('parameters_type_inpaint.txt', 'w') as f:##
    #    print(param_types, file=f)

    #check buffers value
    #with open("model_buffers.txt", "w") as file:#
    #    for name, buffer in model.named_buffers():
    #        file.write(f"Buffer name: {name}\n")
    #        file.write(f"Buffer value:\n{buffer}\n\n")#

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0#shape:1,3,512,512

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)#shape:1,1 ,512,512

    masked_image = image * (mask < 0.5)

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch

def get_edgemap(sampler,uc_full,model,batch,device,model_LGP,cond):
    ###added for sketch
    blocks = [0,1,3]
    image_latent = model.get_first_stage_encoding(model.encode_first_stage(batch["image"]))
    timesteps = torch.zeros((1,), device=device).long()

    noise = 0 * torch.ones(image_latent.shape, device=device)##
    noise = noise.transpose(1,3)     
    
    #get the cond
    if isinstance(cond, dict):
        assert isinstance(uc_full, dict)
        c_in = dict()
        for k in cond:
            if isinstance(cond[k], list):
                c_in[k] = [
                    torch.cat([uc_full[k][i], cond[k][i]])
                    for i in range(len(cond[k]))
                ]
            else:
                c_in[k] = torch.cat([uc_full[k], cond[k]])
    else:
        c_in = torch.cat([uc_full, cond])
    features = extract_features(sampler, image_latent, blocks, c_in, timesteps)
    features = resize_and_concatenate(features, image_latent)

    pred_edge_map = model_LGP(features, noise).unflatten(0, (1, 64, 64)).transpose(3, 1)
    pred_edge_map = model.decode_first_stage(pred_edge_map)
    pred_edge_map = pred_edge_map.cpu()
    pred_edge_map = (pred_edge_map + 1.0) * 127.5
    pred_edge_map = pred_edge_map*(batch["mask"].cpu()<0.5)
    pred_edge_map = pred_edge_map.numpy()
    edge_map = np.clip(pred_edge_map, 0, 255)
    edge_map = edge_map.transpose(0, 2, 3, 1)[0]
    edge_map = Image.fromarray(edge_map.astype(np.uint8))#to Image
    return pred_edge_map,edge_map

#cut the masked area and resize it.
#mask is 1,1,512,512, mask and sketch is 1,3,512,512
#get the index of the most top,bottom,left and right point
#get the lenght of horizon and vertical,top-bottom, right - left
#check whether the top is enough, and add some pixel length/2
#check whether the lest is enough, and add some pixel length/2

def resizer(image,mask,sketch):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    indices = torch.nonzero(mask == 1, as_tuple=False)

    # Finding the most top, bottom, left, and right points
    top = torch.min(indices[:, 2])
    bottom = torch.max(indices[:, 2])
    left = torch.min(indices[:, 3])
    right = torch.max(indices[:, 3])

    # Get the length of horizontal and vertical
    vertical_length = bottom - top + 1
    horizontal_length = right - left + 1
    length = max(vertical_length,horizontal_length)

    # Top padding, assuming we want at least `vertical_length/2` padding
    top = max(0, top - torch.div(length, 2, rounding_mode='floor'))
    # Left padding, assuming we want at least `horizontal_length/2` padding
    left = max(0, left - torch.div(length, 2, rounding_mode='floor'))

    # Adjust bottom and right based on padding added on top and left
    bottom = min(511, top + 2 * length - 1)
    right = min(511, left + 2 * length - 1)

    position = [top,left]
    # Extracting the regions from mask and sketch using the computed indices

    image_small = image[:, :, top:(bottom + 1), left:(right + 1)]
    mask_small = mask[:, :, top:(bottom + 1), left:(right + 1)]
    sketch_small = sketch[:, :, top:(bottom + 1), left:(right + 1)]

    sketch_small = sketch_small.float()
    mask_small = torch.nn.functional.interpolate(mask_small, size=(512, 512), mode='bilinear', align_corners=False)
    image_small = torch.nn.functional.interpolate(image_small, size=(512, 512), mode='bilinear', align_corners=False)
    sketch_small = torch.nn.functional.interpolate(sketch_small, size=(512, 512), mode='bilinear', align_corners=False)

    length = length *2

    return mask_small, image_small, sketch_small, length, position

def mix_img(image, image_small, mask, length, position):
    #resize and mix the image
    #1. I have a tensor 1,3,512,512,resize to the smaller one square 1,3,n,n. 
    #2. make it cover a part of  another tensor which is 13,512,512, the left top point is position = [top,left]. 
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0#shape:1,3,512,512

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)#shape:1,1 ,512,512

    image_small = image_small.cpu().float()

    # Resize samller
    resized = torch.nn.functional.interpolate(image_small, size=(length, length), mode='bilinear', align_corners=False)
    [top,left] = position
    image_small[:, :, top:top+length, left:left+length] = resized
    # mix
    image_m = image_small * mask + (1. - mask) * image

    return image_m

def get_cond(model,batch,num_samples):
    h=512
    w=512
    c = model.cond_stage_model.encode(batch["txt"])

    c_cat = list()
    for ck in model.concat_keys:
        cc = batch[ck].float()
        if ck != model.masked_image_key:
            bchw = [num_samples, 4, h//8, w//8]
            #print("mask before encode",cc.shape)##mask before encode torch.Size([1, 1, 512, 512])
            cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            #print("mask after encode",cc.shape)##mask after encode torch.Size([1, 1, 64, 64])
        else:
            #print("image before encode",cc.shape)##image before encode torch.Size([1, 3, 512, 512])
            cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
            #print("image after Encode",cc.shape)##image after Encode torch.Size([1, 4, 64, 64])
        c_cat.append(cc)
    c_cat = torch.cat(c_cat, dim=1)

    # cond
    cond={"c_concat": [c_cat], "c_crossattn": [c]}
    uc_cross = model.get_unconditional_conditioning(num_samples, "")
    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
    return cond,uc_full

def generate_square_mask(image_size=(512, 512), square_size=64, d_x=0, d_y=0):
    """
    Generates a binary mask with a black square on a white background.
    
    Parameters:
    image_size (tuple): The size of the image for which the mask is being generated.
    square_size (int): The size of the black square mask.
    
    Returns:
    np.ndarray: A binary mask of the specified image size.
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    start_index_1 = d_x+(image_size[0] - square_size) // 2
    end_index_1 = start_index_1 + square_size
    start_index_2 = d_y+(image_size[0] - square_size) // 2
    end_index_2 = start_index_2 + square_size
    mask[start_index_1:end_index_1, start_index_2:end_index_2] = 255
    return mask

def inpaint(sampler,model_LGP, image, mask, prompt, seed, scale, sketch_scale, ddim_steps, middle_number, num_samples=1, w=512, h=512):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h//8, w//8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    #with torch.no_grad():
    with torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h//8, w//8]
                #print("mask before encode",cc.shape)##mask before encode torch.Size([1, 1, 512, 512])
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                #print("mask after encode",cc.shape)##mask after encode torch.Size([1, 1, 64, 64])
            else:
                #print("image before encode",cc.shape)##image before encode torch.Size([1, 3, 512, 512])
                cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                #print("image after Encode",cc.shape)##image after Encode torch.Size([1, 4, 64, 64])
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond={"c_concat": [c_cat], "c_crossattn": [c]}

        #print(cond['c_concat'][0].shape,cond['c_crossattn'][0].shape)##
        

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h//8, w//8]

        pred_edge_map,edge_map = get_edgemap(sampler,uc_full,model,batch,device,model_LGP,cond)
        
        sketch = pred_edge_map
        sketch = torch.from_numpy(sketch)

        ##sample as globle
        image_m = None
        middle_n=10000

        samples_cfg_g, intermediates = sampler.sample(
                model_LGP,
                sketch,##
                middle_n,
                image_m,
                ddim_steps,
                num_samples,
                shape,
                cond,##
                verbose=False,
                eta=1.0,
                unconditional_guidance_scale=scale,
                sketch_scale = sketch_scale,
                unconditional_conditioning=uc_full,##
                x_T=start_code,
        )
        x_samples_ddim_g = model.decode_first_stage(samples_cfg_g)
        
        result = torch.clamp((x_samples_ddim_g+1.0)/2.0,
                                min=0.0, max=1.0)
        result = result.cpu().numpy().transpose(0,2,3,1)
        result = result*255
        result = [Image.fromarray(img.astype(np.uint8)) for img in result]#add twice
        result = [put_watermark(img) for img in result]
        #show the result
        #st.write("Inpainted normal one")
        #for img in result:
        #    st.image(img)

        ##get the batch_small,sketch_small,image_m_s,cond_s
        mask_small, image_small, sketch_small,length, position = resizer(image,mask,sketch)
        masked_small = image_small * (mask_small < 0.5)
        batch_small = {
                "image": repeat(image_small.to(device=device), "1 ... -> n ...", n=num_samples),
                "txt": batch["txt"],
                "mask": repeat(mask_small.to(device=device), "1 ... -> n ...", n=num_samples),
                "masked_image": repeat(masked_small.to(device=device), "1 ... -> n ...", n=num_samples),
                }

        #print(batch['mask'].shape,batch_small['mask'].shape,batch['image'].shape,batch_small['image'].shape,sketch.shape,sketch_small.shape)
        cond_small,uc_full_s = get_cond(model,batch_small,num_samples)
        mask_small, image_m_small, sketch_small,length, position = resizer(result[0],mask,sketch)
        image_m_small = image_m_small.to(device=device)#
        image_m_small = model.get_first_stage_encoding(model.encode_first_stage(image_m_small))


        ##sample as local
        samples_cfg, intermediates = sampler.sample(
                model_LGP,
                sketch_small,#
                middle_number,
                image_m_small,
                ddim_steps,
                num_samples,
                shape,
                cond_small,#
                verbose=False,
                eta=1.0,
                unconditional_guidance_scale=scale,
                sketch_scale = sketch_scale,
                unconditional_conditioning=uc_full_s,#
                x_T=start_code,
        )                 

        x_samples_ddim_s = model.decode_first_stage(samples_cfg)
        ##mix the image
        image_o = mix_img(image, x_samples_ddim_s, mask, length, position)

        #print(samples_cfg.shape,x_samples_ddim.shape)###
        ###torch.Size([1, 4, 64, 64]) torch.Size([1, 3, 512, 512])

        loss = torch.norm(image_o[0] - batch['image'].cpu() ,p=2)###

        result = torch.clamp((image_o+1.0)/2.0,
                                min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0,2,3,1)
        #result, has_nsfw_concept = check_safety(result)#check safety?
        result = result*255

        #print(result.shape)##(1,512,512,3)

        result = [Image.fromarray(img.astype(np.uint8)) for img in result]#add twice
        result = [put_watermark(img) for img in result]

                    

    return result,loss


def run():
    st.title("Stable Diffusion Inpainting")
    
    sampler = initialize_model(sys.argv[1], sys.argv[2])

    #load the LGP
    lgp_path = '/home/tianle/Desktop/stable-diffusion/LGP_10k_2.pt'
    model_LGP = latent_guidance_predictor(output_dim=4, input_dim=6120, num_encodings=9).to('cuda') #where is device
    checkpoint_LGP = torch.load(lgp_path, map_location='cuda')
    model_LGP.load_state_dict(checkpoint_LGP['model_state_dict'])
    model_LGP.eval()

    print(f"{'after initialize all'}: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")####
    print(f"{'after initialize all'}: {torch.cuda.memory_reserved() / 1024**2:.2f} MB allocated")####


    #image = st.file_uploader("Image", ["jpg", "png"])
    folder_A = "/home/tianle/Desktop/stable-diffusion/image_for_inpaint"
            
    prompt = st.text_input("Prompt")
    seed = st.number_input("Seed", min_value=0, max_value=1000000, value=0)
    num_samples = st.number_input("Number of Samples", min_value=1, max_value=64, value=1)
    scale = st.slider("Scale", min_value=0.0, max_value=30.0, value=7.5, step=0.1)
    #sketch_scale = st.slider("Sketch scale(X1000)",min_value=0,max_value=500, value=15, step = 1)
    #sketch_scale = sketch_scale/1000
    middle_number = st.slider("Middle nmuber",min_value=0,max_value=1000,value=600, step = 1)
    ddim_steps = st.slider("DDIM Steps", min_value=0, max_value=50, value=50, step=1)

    sketch_scale = 0.0
    loss = 0
    masks = []
    for i in range(99):
        mask = generate_square_mask(image_size=(512, 512),square_size=np.random.randint(32, 128),d_x= np.random.randint(-100, 100),d_y=np.random.randint(-100, 100))
        masks.append(mask)

    if prompt:
        for num in range(10):
            sketch_scale = sketch_scale + 3 * 10 ** (-6)###
            loss = 0
            numb = 0
            
            for filename in os.listdir(folder_A):
                if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                    # Open the image file
                    img_path = os.path.join(folder_A, filename)
                    image = Image.open(img_path)            
                    
                    #image = Image.open(image)
                    w, h = image.size
                    #print(f"loaded input image of size ({w}, {h})")
                    if max(w, h) > MAX_SIZE:
                        factor = MAX_SIZE / max(w, h)
                        w = int(factor*w)
                        h = int(factor*h)
                    width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
                    image = image.resize((width, height))
                    #print(f"resized to ({width}, {height})")

                    mask = masks[numb]
                    mask = Image.fromarray(mask)
                    numb = numb +1

                    result,loss_tem = inpaint(
                        sampler=sampler,
                        model_LGP=model_LGP,###
                        image=image,
                        mask=mask,
                        prompt=prompt,
                        seed=seed,
                        scale=scale,
                        sketch_scale= sketch_scale,
                        ddim_steps=ddim_steps,
                        middle_number=middle_number,
                        num_samples=num_samples,
                        h=height, w=width
                    )
                    #st.write("Inpainted")
                    #for image in result:
                    #    st.image(image)
                    loss = loss + loss_tem
            with open('sketch_scale_loss.txt', 'a') as file:
                print(f"{'sketch_scale'}: {sketch_scale:.6f} {'loss'}:{loss:.3f}")
                file.write(f"{'sketch_scale'}: {sketch_scale:.6f} {'loss'}:{loss:.1f}\n")


if __name__ == "__main__":
    run()
