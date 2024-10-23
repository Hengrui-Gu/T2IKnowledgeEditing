import os
import random
import json

from PIL import Image
import numpy as np

import torch
import open_clip
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionSafetyChecker
from scipy import stats

from directory_path import *

from utils import load_json,save_json

class SafteyChecker(StableDiffusionSafetyChecker):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, clip_input, images):
        has_nsfw_concepts = [False for _ in range(len(images))]
        return images, has_nsfw_concepts

    def forward_onnx(self, clip_input: torch.FloatTensor, images: torch.FloatTensor):
        has_nsfw_concepts = [False for _ in range(len(images))]
        return images, has_nsfw_concepts

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def read_dataset(dataset_path):

    with open(dataset_path,"r+") as f:
        target_dataset = json.loads(f.read())
    return target_dataset

def generate_with_seed(sd_pipeline, prompts, seed, output_path="./", image_params="", save_image=True):
    set_seed(seed)
    outputs = []
    for prompt in prompts:
        print(prompt)
        image = sd_pipeline(prompt)['images'][0]

        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except FileExistsError:
                pass

        if image_params != "":
            image_params = "_" + image_params

        image_name = f"{output_path}/seed_{seed}{image_params}.png"
        if save_image:
            image.save(image_name)
        print("Saved to: ", image_name)
        outputs.append((image, image_name))

    if len(outputs) == 1:
        return outputs[0]
    return outputs


def get_CLIP_scores(preprocess_val, tokenizer, model, output_path, golden_prompt,  seed, device):
    # return the cosine similarity
    generated_image_path = f"{output_path}/seed_{seed}.png"
    try:
        image = Image.open(generated_image_path)
    except:
        print("**** This image does not exists !!! ****")
        print(generated_image_path)

    image = preprocess_val(image).unsqueeze(0).to(device)
    text = tokenizer([golden_prompt]).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        Sim_cosine = image_features @ text_features.T

    return Sim_cosine[0][0].item()

def wrap_stats(data):
    
    data = np.array(data)
    return {
        "mean":data.mean(),
        "1sigma":data.mean()-data.std(ddof=1),
        "2sigma":data.mean()-2*data.std(ddof=1),
        "3sigma":data.mean()-3*data.std(ddof=1),
        "+1sigma":data.mean()+data.std(ddof=1),
        "+2sigma":data.mean()+2*data.std(ddof=1),
        "+3sigma":data.mean()+3*data.std(ddof=1),
    }

def Generate_Threshold_samples(args,dataset) -> str:
    
    print(f'Generate images conditioned on golden prompts for calculating adaptive-clip-thresholds --model: {args.model} -dataset: {args.dataset} - seed num: {args.evaluation_seeds}')
    print(args)
    set_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    sd_model_name = f"{MODEL_DIR}/{args.model}"
    sd_pipeline = StableDiffusionPipeline.from_pretrained(sd_model_name, use_auth_token=HF_TOKEN)
    sd_pipeline.safety_checker = SafteyChecker(sd_pipeline.safety_checker.config)
    sd_pipeline = sd_pipeline.to(device)

    images_path = f'{EVA_IMAGES_DIR}/{args.model}/{args.dataset}'
    os.makedirs(images_path, exist_ok=True)
    num_seeds = args.evaluation_seeds

    # only the preprocess_val is useful for processing raw images : CenterCrop:(224,224) -> Layer-Normalize(mean,std)

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            'ViT-bigG-14',pretrained = OPENCLIP_MODEL)
    model = model.to(device)
    
    tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

    # initialize thresholds stats
    stats_path = f'{images_path}/statistics.json'

    if os.path.exists(stats_path):
        stats = load_json(stats_path)
    else:
        stats = {}

    for row in dataset:
        
        for edit in row:
            
            tag = edit['tag']
            output_path = f'{images_path}/{tag}'

            if tag not in stats.keys():
                stats[tag] = {}
        
            for metric, prompts in edit['test_prompts'].items():
                
                for prompt in prompts:
                
                    prompt_key = 'test'
                    print(f'Generation on {prompt[prompt_key]}')
                    prompt_path = f'{output_path}/{prompt[prompt_key]}'

                    seed_stats = {}
                    if prompt['test'] in stats[tag].keys():
                        seed_stats = stats[tag][prompt['test']]
                        
                    for seed in range(num_seeds):
                        
                        image_path = f'{prompt_path}/seed_{seed}.png'

                        if not os.path.exists(image_path):
                            generate_with_seed(sd_pipeline, [prompt['test_eval']], seed,
                                            output_path=prompt_path)
                        
                        if f'seed_{seed}' not in seed_stats.keys():
                            Sim_cosine = get_CLIP_scores(preprocess_val, tokenizer, model, prompt_path, prompt['test_eval'], seed, device)
                            seed_stats[f'seed_{seed}'] = Sim_cosine
                    
                    stats[tag][prompt['test']] = seed_stats

    save_json(stats_path,stats)    
                   
    del model                
    del sd_pipeline

    return stats_path

def Calculate_CLIP_Thresholds(args, stats_path, clip_threshold_path ,dataset):
    
    clip_score = load_json(stats_path)
    num_seeds = args.evaluation_seeds
    
    thresholds = {}
    for row in dataset:

        for edit in row:
            tag = edit['tag']
            thresholds[tag] = {}
            instance_dict = clip_score[tag]

            for metric, prompts in edit['test_prompts'].items():
                
                for prompt in prompts:
                    prompt_dict = instance_dict[prompt['test']]
                    thresholds_over_seeds = []

                    for seed in range(num_seeds):
                        thresholds_over_seeds.append(prompt_dict[f'seed_{seed}'])
                    
                    result = wrap_stats(thresholds_over_seeds)
                    thresholds[tag][prompt['test']] = result
                    
    save_json(clip_threshold_path, thresholds)

    return thresholds

if __name__ == "__main__":
    pass
