import os, sys

# set cach path
cache_path = '/mnt/data/hzy_data/hf_file'
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_path, 'datasets')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_path, 'models')

from lm_eval import evaluator
from lm_eval.models import huggingface
from lm_eval.tasks import get_task_dict
from lm_eval.api.registry import TASK_REGISTRY


def evaluate_model(model_name, task_names, device="cuda:0", batch_size=8):
    """
    evaluate model by lm-evaluation-harness 
    HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf --model_args pretrained="gpt2" --tasks hellaswag --device cuda:0 --batch_size 8

    Args:
        model_name (str): HuggingFace model name or local path（eq "facebook/opt-6.7b"）
        task_names (list): task list（eg. ["boolq", "hellaswag"]）
        device (str): device（eq"cuda:0" or "cpu"）
        batch_size (int): 
    """
    # Load huggingface model
    model = huggingface.HFLM(
        pretrained=model_name,
        device=device,
        # cache_dir=cache_path,
        batch_size=batch_size
    )
    results = evaluator.simple_evaluate(
    model=model,
    tasks=task_names,
    num_fewshot=0,
    batch_size=4,
    device="cuda:0"
)

    # print result
    print("\nevalate result:")
    for task_name, task_results in results["results"].items():
        print(f"\ntask: {task_name}")
        print(f"\nresults: {task_results}")

    return results

if __name__ == "__main__":
    from datasets import config
    from transformers.utils.hub import TRANSFORMERS_CACHE

    print(f"dataset cache: {config.HF_DATASETS_CACHE}")
    print(f"model path cache: {TRANSFORMERS_CACHE}")

    # model_name = "gpt2"
    # model_name = "facebook/opt-6.7b"
    model_name = "meta-llama/Llama-2-7b"  
    tasks = ["hellaswag, winogrande, boolq, openbookqa, piqa"]     # eval_task 
    evaluate_model(model_name, tasks, device="cuda:0", batch_size=4)