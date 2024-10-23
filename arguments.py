import argparse
import numpy

def parse_args():
    parser = argparse.ArgumentParser(
        prog='Diffusion Bootstrap Editor',
        description='A script for editing text-to-image diffusion model using different algorithm and conducting bootstrap evaluation')
    parser.add_argument('--dataset', default='CAKE', choices=["CAKE","TIMED","RoAD"])
    parser.add_argument('--alg', default='MPE', choices=['MPE','baseline'])
    parser.add_argument('--model', default='stable-diffusion-v1-4')
    parser.add_argument('--clip_model', default='local_checkpoints/clip-vit-large-patch14-336')
    parser.add_argument('--edit_mode',default='single',choices= ['single','multiple'])
    parser.add_argument('--edit_batch_size', type=int, default=25,choices=[10,25,50,110])

    parser.add_argument('--generation_seeds', type=int, default=10)
    parser.add_argument('--evaluation_seeds', type=int, default=50)
    parser.add_argument('--threshold_type', type=str, default='2sigma', choices=['3sigma','2sigma','1sigma','mean','+1sigma','+2sigma','+3sigma'])
    parser.add_argument('--only_evaluation', action='store_true')

    # arguments for ICL-based MPE

    parser.add_argument('--LLM', default='gpt-3.5-turbo-0125')
    parser.add_argument('--Openai_API_key', type=str)
    parser.add_argument('--prompt', default='MPE_prompt')

    return parser.parse_args()


