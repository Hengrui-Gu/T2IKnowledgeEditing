import subprocess
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionSafetyChecker
from directory_path import *
from utils import *
import os
import time
from openai import OpenAI
import httpx

def MPE_single_editing(args, dataset):
    
    client = OpenAI(
        api_key = args.Openai_API_key,
    )

    def call_gpt(cur_prompt, stop=['\n\n']):
        
        ans = client.chat.completions.create(
            model=args.LLM,
            messages=[{"role": "user", "content": cur_prompt}],
            temperature=0,
            stop=stop,
            max_tokens = 300
        )

        returned = ans.choices[0].message.content
        stop = ans.choices[0].finish_reason

        return returned, stop

    def load_txt(path):
        with open(path, 'r+') as f:
            file = f.read()
        return file

    def parse_output(raw_output):
        have_result = False
        rows = raw_output.strip().split('\n')
        final_output = ""

        if len(rows) == 2:
            first_row = rows[0]
            flag = first_row.split(': ')[-1].strip()[:-1]
    
            if flag in ['Yes','No']:
                final_row = rows[-1]
    
                if final_row.startswith("Output: "):
                    final_list = final_row.split(": ")
                    if len(final_list) == 2:
                        final_output = final_list[-1]
                        have_result = True
        
        return final_output, flag, have_result


    # In-context based MPE editing
    model_name = args.LLM

    prompt_path = f"{args.prompt}.txt"
    mapping_prompt = load_txt(prompt_path)

    PLM_dir = f'{PLM_RESULTS}/{args.dataset}/{args.LLM}'

    PLM_file_path = f'{PLM_dir}/{args.prompt}.json'
    PLM_stat_file_path  = f'{PLM_dir}/{args.prompt}_stats.json'

    dataset = load_json(f'dataset/refined_dataset/{args.dataset}.json')
    
    filtered_prompts = []
    metric_acc = {}

    for row in dataset:

        row_filtered_prompts = {}
        edits = []
        for ins in row:

            edits.append({'prompt':ins['prompt'], 'target':ins['target']})
            
            for metric, prompts in ins['test_prompts'].items():
                
                if metric not in metric_acc.keys():
                    metric_acc[metric] = {'positive':0 , 'sum':0}
                
                for prompt in prompts:
                    
                    metric_acc[metric]['sum'] += 1
                    input_text = prompt['test']

                    print(f'call gpt for prompt : {input_text}')      
                    
                    Router_history = []
                    Router_flag = False
                    for edit in edits:
                        # edit-specific suffix
                        suffix_prompt = f"source concept: {edit['prompt']}.\n" + f"target concept: {edit['target']}.\n" 

                        llm_prompt = mapping_prompt + "\n\n" + "Input: " + f"{input_text}\n" + suffix_prompt
                        output, have_stop = call_gpt(llm_prompt)
                
                        if not have_stop == 'stop':
                            break

                        output_text, flag , have_result = parse_output(output)
                        
                        if not have_result:
                            break

                        if flag == 'No':
                            break
                        else:
                            input_text = output_text

                    if input_text == prompt['test_eval']:
                        metric_acc[metric]['positive'] += 1
                        key1 = 'positive'
                        key2 = 'sum'
                        print(f'prompt: {input_text} - {metric} -- {metric_acc[metric][key1]} / {metric_acc[metric][key2]} ({metric_acc[metric][key1]/metric_acc[metric][key2]})')

                    # row_filtered_prompts[prompt['test']] = input_text
                    row_filtered_prompts[prompt['test']] = input_text

        print(row_filtered_prompts)
        filtered_prompts.append(row_filtered_prompts)
    
    save_json(PLM_file_path, filtered_prompts)
    save_json(PLM_stat_file_path, metric_acc)

def PLM_image_generations(args,dataset,refined_prompts):
    
    # image_dir = f'{IMAGES_DIR}/{args.alg}/{args.dataset}/{args.model}/{args.LLM}/{args.prompt}'
    image_dir = f'{IMAGES_DIR}/{args.alg}/{args.dataset}/{args.model}/{args.prompt}'
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
    time_arr = []
    # if image with the same name already exists, just re-generate the target image and overwrite it.
    for row, map_prompts in zip(dataset,refined_prompts):
        
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
                            start_time = time.time()
                            generate_with_seed(sd_pipeline, [map_prompts[raw_prompt]], seed, output_path)
                            end_time = time.time()
                            print(f'{end_time - start_time} s')
                            time_arr.append(end_time - start_time)

    print(time_arr)
    print(f'generation will cost average: {np.mean(time_arr)} s')
    del sd_pipeline
    return image_dir

def PLM_multiple_image_generations(args,dataset,refined_prompts):
    
    image_dir = f'{IMAGES_DIR}/{args.alg}/{args.dataset}/{args.model}/{args.prompt}/batch:{args.edit_batch_size}'
    path_makedir(image_dir)
    if args.only_evaluation:
        return image_dir

    print(f'multiple edit on bsz:{args.edit_batch_size} Generate images for alg: {args.alg} using sd model : {args.model}')
    
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
                            generate_with_seed(sd_pipeline, [refined_prompts[raw_prompt]], seed, output_path)

    del sd_pipeline
    return image_dir


if __name__ == '__main__':
    pass