# T2IKnowledgeEditing

This is the official repo of our paper [Pioneering Reliable Assessment in Text-to-Image Knowledge Editing: Leveraging a Fine-Grained Dataset and an Innovative Criterion](https://arxiv.org/abs/2409.17928).

This paper has been accepted by **EMNLP24 Findings**

## Content
[1. Release of Pre-calculated CLIP Thresholds](#release-of-pre-calculated-clip-thresholds)

[2. Commands](#commands)

[3. Thanks](#thanks)

[4. Citation](#citation)

## Release of Pre-calculated CLIP Thresholds

We have calculated and released evaluation thresholds for our settings (50 random seeds) on T2I editing datasets (CAKE, RoAD, and TIME Dataset) in `eva_clip_thresholds/stable-diffusion-v1-4/{dataset_name}/seed-50.json`. These pre-cached threshold files enable users to skip the extra warm-up stage and proceed directly with edited image evaluation.

## Commands

```
# Run MPE
CUDA_VISIBLE_DEVICES=0 python T2IDiffusion-edit.py --dataset CAKE --alg MPE --generation_seeds 10 --evaluation_seeds 50 --threshold_type 2sigma --LLM gpt-3.5-turbo-0125 -Openai_API_key YOUR-API-KEY


# Run "Base"
CUDA_VISIBLE_DEVICES=0 python T2IDiffusion-edit.py --dataset CAKE --alg baseline --generation_seeds 10 --evaluation_seeds 50 --threshold_type 2sigma
```

## Thanks

We utilize two outstanding T2I knowledge editing datasets in this repo: [RoAD](https://github.com/technion-cs-nlp/ReFACT) and the [TIME Dataset](https://github.com/bahjat-kawar/time-diffusion). We sincerely thank the creators for their contributions.


## Citation

If our paper or code is helpful to you, please consider citing our work:

```
@misc{gu2024pioneeringreliableassessmenttexttoimage,
      title={Pioneering Reliable Assessment in Text-to-Image Knowledge Editing: Leveraging a Fine-Grained Dataset and an Innovative Criterion}, 
      author={Hengrui Gu and Kaixiong Zhou and Yili Wang and Ruobing Wang and Xin Wang},
      year={2024},
      eprint={2409.17928},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.17928}, 
}
```
