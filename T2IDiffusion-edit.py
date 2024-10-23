import os
import time
import random

from PIL import Image
import numpy as np
import pandas as pd
import torch

from arguments import parse_args
from utils import *
from directory_path import *
from init_Bootstrap_evaluation import Generate_Threshold_samples,Calculate_CLIP_Thresholds
from alg_MPE import PLM_image_generations,PLM_multiple_image_generations,MPE_single_editing
from alg_baseline import baseline_generations

os.environ['NUMEXPR_MAX_THREADS'] = '16'

def main():

    args = parse_args()
    configs = load_json(CONFIG_PATH)

    #  load different dataset
    dataset = unify_dataset_format(configs, args.dataset)

    #  save the refined dataset
    save_json(f'dataset/refined_dataset/{args.dataset}.json',dataset)

    #  The warm-up stage of the Adaptive-CLIP-Threshold-Criterion
    clip_threshold_dir = f'{EVA_CLIP_SCORE_DIR}/{args.model}/{args.dataset}'
    os.makedirs(clip_threshold_dir, exist_ok = True)
    clip_threshold_path = f'{clip_threshold_dir}/seed-{args.evaluation_seeds}.json'
    
    if not os.path.exists(clip_threshold_path):
        stats_path = Generate_Threshold_samples(args,dataset)
        torch.cuda.empty_cache()
        clip_thresholds = Calculate_CLIP_Thresholds(args, stats_path, clip_threshold_path, dataset)
    else:
        clip_thresholds = load_json(clip_threshold_path)
        print(f'\n loading existing thresholds file for ---- {args.dataset} ---- \n')

    image_dir = ''
    
    #  perform different editing techniques  
    if args.alg == 'MPE':
        
        if args.edit_mode == 'single':

            PLM_dir = f'{PLM_RESULTS}/{args.dataset}/{args.LLM}'
            os.makedirs(PLM_dir, exist_ok=True)
            
            PLM_result_path = f'{PLM_dir}/{args.prompt}.json'
            PLM_stats_path = f'{PLM_dir}/{args.prompt}_stats.json'

            # pre-process raw prompts from editing datasets
            if not os.path.exists(PLM_result_path):
                MPE_single_editing(args,dataset)

            # Editing results
            PLM_stats = load_json(PLM_stats_path)
            for k,v in PLM_stats.items():
                print(f'MPE refined prompt Acc : {k} - {v}')

            refined_prompts = load_json(PLM_result_path)
            
            # generate edited images conditioend on edited prompts
            image_dir = PLM_image_generations(args,dataset,refined_prompts)
        
        # elif args.edit_mode == 'multiple':
        #     PLM_result_path = f'{PLM_RESULTS}/a_multiple_edits/{args.LLM}/{args.dataset}/{args.prompt}-{args.edit_batch_size}.json'

        #     if not os.path.exists(PLM_result_path):
        #         print(f'Lack the refined prompts produced by {args.LLM} on {args.dataset} using {args.prompt} with bsz = {args.edit_batch_size}')
        #         assert False
            
        #     PLM_stats_path = f'{PLM_RESULTS}/a_multiple_edits/{args.LLM}/{args.dataset}/{args.prompt}-{args.edit_batch_size}_stats.json'
        #     PLM_stats = load_json(PLM_stats_path)

        #     for k,v in PLM_stats.items():
        #         print(f'PLM refined prompt Acc : {k} - {v}')

        #     refined_prompts = load_json(PLM_result_path)
        #     image_dir = PLM_multiple_image_generations(args,dataset,refined_prompts)
            
    elif args.alg == 'baseline':
        
        image_dir = baseline_generations(args,dataset)
        
    torch.cuda.empty_cache()

    #  conduct evaluation based on Adaptive clip threshold for generated images using corresponding editing method

    stats_path = calculate_clip_score(args,dataset,image_dir)
    
    logs = calculate_metrics(args, dataset, stats_path, clip_thresholds)

    print(f'-- Results of Editing {args.model} by {args.alg} on seeds {args.generation_seeds}-- ')

    for metric, results in logs.items():
        key1 = 'mean'
        key2 = 'std'
        print(f'metric: {metric} --- mean: {results[key1]} std: {results[key2]}')

if __name__ == '__main__':
    main()
