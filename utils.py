import pandas as pd
import numpy as np
import json
from copy import deepcopy
from directory_path import *
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionSafetyChecker
import open_clip
import os
import random
import torch
from PIL import Image
import time

def load_json(path):
    with open(path,'r+') as f:
        file = json.loads(f.read())
    return file

def save_json(path,file) -> None:

    # create corresponding directory for to be saved json file
    path_list = path.split('/')
    root_path = ''
    for element in path_list[:-1]:
        root_path = root_path + element +'/' 
                        
    if not os.path.exists(root_path):
        try:
            os.makedirs(root_path)
        except FileExistsError:
            pass

    # save file
    with open(path,'w+') as f:
        f.write(json.dumps(file))

def path_makedir(path) -> None:

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

class SafteyChecker(StableDiffusionSafetyChecker):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, clip_input, images):
        has_nsfw_concepts = [False for _ in range(len(images))]
        return images, has_nsfw_concepts

    def forward_onnx(self, clip_input: torch.FloatTensor, images: torch.FloatTensor):
        has_nsfw_concepts = [False for _ in range(len(images))]
        return images, has_nsfw_concepts
    
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

def unify_dataset_format(configs,dataset_name):
    
    path = configs['path'][dataset_name]
    processed_dataset = []

    if dataset_name == 'CAKE':
        dataset = load_json(path)

        for edit, compo_edit in zip(dataset['single_edit'], dataset['composite_edit']):
            
            # single edit
            entry = []
            edit_prompt = edit['edit_prompt'].format(edit['entity'])
            tag = edit_prompt
            target = edit['target']

            # prompt oracle target tag
            candidates_keys = ['role_type','edit_prompt','entity','target','oracle']
            generation_keys = ['generality_a','generality_b','specificity']

            first_edit = {'tag':tag,'prompt':edit_prompt}
            for key in candidates_keys:
                first_edit[key] = edit[key]
            
            first_edit['test_prompts'] = {}
            first_edit['test_prompts']['efficacy'] = [{'test':edit_prompt,'test_oracle':edit['oracle'],'test_eval':edit['target']}]

            for key in generation_keys:
                if key in edit.keys():
                    first_edit['test_prompts'][key] = edit[key]
            
            entry.append(first_edit)

            # composite edit
            assert edit_prompt == compo_edit['edits'][0]['edit_prompt'].format(compo_edit['edits'][0]['entity'])
            assert target == compo_edit['edits'][0]['target']

            edit_prompt = compo_edit['edits'][1]['edit_prompt'].format(compo_edit['edits'][1]['entity'])
            second_edit = {'tag':f'composite/{tag}', 'prompt':edit_prompt}
            for key in candidates_keys:
                second_edit[key] = compo_edit['edits'][1][key]
            
            second_edit['test_prompts'] = {}
            second_edit['test_prompts']['compositionality'] = compo_edit['compositionality']
            entry.append(second_edit)
            
            processed_dataset.append(entry)

    elif dataset_name == 'TIMED':
        
        valid_set = pd.read_csv(path)
        
        repeat_list = []
        repeat_dict = {}
        for idx, raw_row in valid_set.iterrows():
            row = dict()
            for k,v in raw_row.items():
                row[k.lower()] = v.lower()

            if row['old'] not in repeat_list:
                repeat_list.append(row['old'])
            else:
                if row['old'] not in repeat_dict.keys():
                    repeat_dict[row['old']] = [row['old']+'_0',row['old']+'_1']
                else:
                    key = 'old'
                    repeat_dict[row['old']].append(row['old']+f'_{len(repeat_dict[row[key]])}')
            
        for idx, raw_row in valid_set.iterrows():
            row = dict()
            for k,v in raw_row.items():
                row[k.lower()] = v.lower()

            entry = []
            edit = {}
            # prompt oracle target
            edit['tag'] = row['old']
            if row['old'] in repeat_dict.keys():
                edit['tag'] = repeat_dict[row['old']][0]
                repeat_dict[row['old']] = repeat_dict[row['old']][1:]
            
            edit['prompt'] = row['old']
            edit['oracle'] = row['new']
            edit['target'] = row['new']
            edit['old'] = row['old']

            edit['test_prompts'] = {}
            # efficacy metric
            metric = 'efficacy'
            edit['test_prompts'][metric] = [{'test':row['old'],'test_oracle':row['new'],'test_eval':row['new']}]

            # positive metric  -> generality
            metric = 'positive'
            members = ['positive{}','gt{}','gt{}']
            mapping = ['test','test_oracle','test_eval']

            edit['test_prompts'][metric] = [{key:row[raw_key.format(idx)] for raw_key,key in zip(members,mapping)} for idx in range(1,6)]

            # negative metric  -> specificity
            metric = 'negative'
            members = ['negative{}','negative{}','negative{}']
            mapping = ['test','test_oracle','test_eval']

            edit['test_prompts'][metric] = [{key:row[raw_key.format(idx)] for raw_key,key in zip(members,mapping)} for idx in range(1,6)]

            entry.append(edit)
        
            processed_dataset.append(entry)
            
    elif dataset_name == 'RoAD':
        
        valid_set = pd.read_csv(path)
        
        for idx, raw_row in valid_set.iterrows():
            row = dict()
            for k,v in raw_row.items():
                row[k.lower()] = v.lower()

            entry = []
            edit = {}
            # prompt oracle target
            edit['tag'] = row['prompt']
            edit['prompt'] = row['prompt']
            edit['oracle'] = row['oracle']
            edit['target'] = row['new']
            edit['old'] = row['old']

            edit['test_prompts'] = {}
            # efficacy metric
            metric = 'efficacy'
            edit['test_prompts'][metric] = [{'test':row['prompt'],'test_oracle':row['oracle'],'test_eval':row['new']}]

            # positive metric  -> generality
            metric = 'positive'
            members = ['positive{}','positive_oracle{}','positive_new{}']
            mapping = ['test','test_oracle','test_eval']

            edit['test_prompts'][metric] = [{key:row[raw_key.format(idx)] for raw_key,key in zip(members,mapping)} for idx in range(1,6)]

            # negative metric  -> specificity
            metric = 'negative'
            members = ['negative{}','negative{}','negative{}']
            mapping = ['test','test_oracle','test_eval']

            edit['test_prompts'][metric] = [{key:row[raw_key.format(idx)] for raw_key,key in zip(members,mapping)} for idx in range(1,6)]

            entry.append(edit)
        
            processed_dataset.append(entry)

    return processed_dataset


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

def calculate_clip_score(args,dataset,image_dir):
    
    print(f'Use openclip to calculate efficacy/generality/specificity for {args.alg}')

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            'ViT-bigG-14',pretrained = OPENCLIP_MODEL)
    model = model.to(device)
    
    tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    
    num_image_seeds = args.generation_seeds

    stats_path = f'{image_dir}/statistics.json'
    stats = {}
    if os.path.exists(stats_path):
        stats = load_json(stats_path)

    for row in dataset:

        for edit in row:
            tag = edit['tag']
            root_path = f'{image_dir}/{tag}'

            if tag not in stats.keys():
                stats[tag] = {}

            for metric, prompts in edit['test_prompts'].items():
                
                for prompt in prompts:
                    
                    prompt_key = 'test'
                    print(f'Cal clip score for prompt {prompt[prompt_key]} generated by {args.alg}')
                    prompt_path = f'{root_path}/{prompt[prompt_key]}'

                    seed_stats = {}
                    if prompt['test'] in stats[tag].keys():
                        seed_stats = stats[tag][prompt['test']]
                        
                    for seed in range(num_image_seeds):
                        
                        if f'seed_{seed}' not in seed_stats.keys():
                            Sim_cosine = get_CLIP_scores(preprocess_val, tokenizer, model, prompt_path, prompt['test_eval'], seed, device)
                            seed_stats[f'seed_{seed}'] = Sim_cosine
                    
                    stats[tag][prompt['test']] = seed_stats

    save_json(stats_path,stats)
    return stats_path


def calculate_metrics(args, dataset, stats_path, clip_thresholds):
    
    num_seeds = args.generation_seeds
    stats = load_json(stats_path)

    metric_results = {}
    for row in dataset:
        
        for edit in row:
            
            tag = edit['tag']
            for metric, prompts in edit['test_prompts'].items():
                
                if metric not in metric_results.keys():
                    metric_results[metric] = {"correct":np.zeros((num_seeds)),"sum":0}

                for prompt in prompts:
    
                    prompt_key = 'test'
                    test_prompt = prompt[prompt_key]
                    metric_results[metric]['sum'] += 1
                    seed_stats = stats[tag][test_prompt]

                    threshold = clip_thresholds[tag][test_prompt][args.threshold_type]
                    clip_score = [seed_stats[f'seed_{seed}'] for seed in range(num_seeds)]
                    
                    metric_results[metric]["correct"] += np.array(clip_score) >= threshold
    
    
    logs = {}
    
    if args.alg =='PLM':
        log_path = f'{LOG_DIR}/{args.dataset}/{str(args.edit_batch_size)+"bsz/" if args.edit_mode == "multiple" else ""}{args.alg}_{args.prompt}_{args.LLM}_type:{args.threshold_type}_eva:{args.evaluation_seeds}.log'
    elif args.alg =='baseline':
        log_path = f'{LOG_DIR}/{args.dataset}/{str(args.edit_batch_size)+"bsz/" if args.edit_mode == "multiple" else ""}{args.alg}_type:{args.threshold_type}_eva:{args.evaluation_seeds}.log'

    for metric, results in metric_results.items():
        
        results['correct'] = results['correct']/results['sum']
        logs[metric] = {"mean": results['correct'].mean(), "std": results['correct'].std()}

    save_json(log_path,logs)

    return logs
    



                            
                            
                            
                        
                        

    