#download Imagenet
python ./downloader.py \
    -data_root data_root_folder\
    -number_of_classes 120 \
    -images_per_class 8

#generate edge by pidinet from image
python main.py --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir oneedge --datadir oneimage --dataset image_dataset --evaluate trained_models/table7_pidinet.pth --evaluate-converted

python Evaluate_LGP.py --caption "woman" --noise_strength 0.3 --image_path testimage.jpg --LGP_path SDv1.5-trained_LGP.pt --vae runwayml/stable-diffusion-v1-5 --unet runwayml/stable-diffusion-v1-5 --device cuda

python train_LGP.py --dataset_dir imagenet_images --edge_maps_dir edge_dir --batch_size 5 --LGP_path /SDv1.5-trained_LGP.pt --epochs 6 --lr 0.0001 --device cuda --vae runwayml/stable-diffusion-v1-5 --unet runwayml/stable-diffusion-v1-5

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

python try.py $MODEL_FLAGS --classifier_scale 0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS

of try.py
python try.py $MODEL_FLAGS --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS

#command of train_LGP.py
#if epochs more than 4, it cant be faster anymore
python train_LGP_block2.py --dataset_dir imagenet_images_1k --edge_maps_dir edge_dir_1k --batch_size 1 --LGP_path trained_LGP_1k_7 --epochs 4 --lr 0.0001 --device cuda --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --model_path models/256x256_diffusion.pt --batch_size2 2

#command for evaluate the LGP
python Evaluate_LGP.py --timesteps 900 --image_path testimage.jpg --LGP_path trained_LGP_1k_new.pt --device cuda --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --model_path models/256x256_diffusion.pt

#command for the the gradient
python LGP_evaluate_2.py --timesteps 600 --image_path original_dog.jpg --edge_path edge_map.jpg --LGP_path trained_LGP_1k_3.pt --device cuda --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --model_path models/256x256_diffusion.pt


###guided diffusion 256
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS

##guided diffusion 512
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 512 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 4.0 --classifier_path models/512x512_classifier.pt --model_path models/512x512_diffusion.pt $SAMPLE_FLAGS

##guided diffusion 256 with gradient
python classifier_sample_guided.py --classifier_scale 0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt --edge_path edge_map.jpg --LGP_path trained_LGP_90k_4.pt --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True

python classifier_sample_guided.py --classifier_scale 0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt --edge_path edge_map.jpg --LGP_path trained_LGP_1k_4.pt --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True

##128
python classifier_sample_guided.py --classifier_scale 0.5 --classifier_path models/128x128_classifier.pt --model_path models/128x128_diffusion.pt --edge_path edge_map.jpg --LGP_path trained_LGP_90k_4.pt --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True


# Train
#sd v1.5
python train_LGP.py --dataset_dir imagenet_images_90k --edge_maps_dir edge_dir_90k --batch_size 16 --LGP_path LGP_90k_one --epochs 4 --lr 0.0001 --device cuda --vae runwayml/stable-diffusion-v1-5 --unet runwayml/stable-diffusion-v1-5

#train for inpainting
python train_LGP.py --dataset_dir imagenet_images_10k --edge_maps_dir edge_dir_10k --batch_size 8 --LGP_path LGP_10k_2 --epochs 1 --lr 0.0001 --device cuda --vae runwayml/stable-diffusion-inpainting --model_path models/ldm/inpainting_big/sd-v1-5-inpainting.ckpt --configs configs/stable-diffusion/v1-inpainting-inference.yaml

#train by real image
python train_LGP_sketch.py --dataset_dir image_real_more --edge_maps_dir sketch_rendered/width-5 --batch_size 1 --LGP_path LGP_real_5k_1p_5 --epochs 1 --lr 0.0001 --device cuda --vae runwayml/stable-diffusion-inpainting --model_path models/ldm/inpainting_big/sd-v1-5-inpainting.ckpt --configs configs/stable-diffusion/v1-inpainting-inference.yaml

#real
python train_LGP_sketch.py --dataset_dir image_real_more --edge_maps_dir sketch_rendered/width-5 --batch_size 20 --LGP_path LGP_real_5k_12p_5 --epochs 12 --lr 0.0001 --device cuda --model_path models/ldm/inpainting_big/sd-v1-5-inpainting.ckpt --configs configs/stable-diffusion/v1-inpainting-inference.yaml

#evaluate
python Evaluate_LGP.py --caption "human ride the horse" --noise_strength 0.4 --image_path 00008.png --LGP_path LGP_90k_1.pt --vae runwayml/stable-diffusion-inpainting --device cuda --model_path models/ldm/inpainting_big/sd-v1-5-inpainting.ckpt --configs configs/stable-diffusion/v1-inpainting-inference.yaml

# stable diffusion inpainting
streamlit run scripts/inpaint_st.py -- configs/stable-diffusion/v1-inpainting-inference.yaml models/ldm/inpainting_big/sd-v1-5-inpainting.ckpt

##streamlit need 1.23.0

