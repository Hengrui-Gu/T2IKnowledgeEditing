import subprocess
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionSafetyChecker
from directory_path import *
from utils import *
import os

def baseline_generations(args,dataset):
    
    image_dir = f'{IMAGES_DIR}/{args.alg}/{args.dataset}/{args.model}'
    path_makedir(image_dir)

    # whether to skip the generation stage
    if args.only_evaluation:
        return image_dir

    print(f'Generate images for alg: {args.alg} using sd model : {args.model}')

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    sd_model_name = f"{MODEL_DIR}/{args.model}"
    sd_pipeline = StableDiffusionPipeline.from_pretrained(sd_model_name, use_auth_token=HF_TOKEN)
    sd_pipeline.safety_checker = SafteyChecker(sd_pipeline.safety_checker.config)
    sd_pipeline = sd_pipeline.to(device)

    # if image with the same name already exists, just re-generate the target image and overwrite it.
    for row in dataset:
        
        for ins in row:
            tag = ins['tag']
            
            for metric, prompts in ins['test_prompts'].items():
                
                for prompt in prompts:
                    raw_prompt = prompt['test']
                    output_path = f'{image_dir}/{tag}/{raw_prompt}'

                    for seed in range(args.generation_seeds):
                        seed_path = f'{output_path}/seed_{seed}.png'
                        
                        if os.path.exists(seed_path):
                            print('This image exists!')
                        else:
                            generate_with_seed(sd_pipeline, [raw_prompt], seed, output_path)

    del sd_pipeline
    return image_dir