
# Run T2IKnowledgeEditing

# Run MPE
CUDA_VISIBLE_DEVICES=0 python T2IDiffusion-edit.py --dataset CAKE --alg MPE --generation_seeds 10 --evaluation_seeds 50 --threshold_type 2sigma --LLM gpt-3.5-turbo-0125 -Openai_API_key YOUR-API-KEY


# Run "Base"
CUDA_VISIBLE_DEVICES=0 python T2IDiffusion-edit.py --dataset CAKE --alg baseline --generation_seeds 10 --evaluation_seeds 50 --threshold_type 2sigma