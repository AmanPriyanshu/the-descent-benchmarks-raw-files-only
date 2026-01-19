# main.py
from typing import Callable, Dict, Any
from dataset_utils import get_dataset_handler

def run_benchmark(dataset_name: str, run_inference_fn: Callable) -> Dict[str, Any]:
    print(f"{'='*80}")
    print(f"Running benchmark: {dataset_name}")
    print(f"{'='*80}")
    
    handler = get_dataset_handler(dataset_name)
    
    contexts, instructions = handler.get_contexts_and_instructions()
    ids = handler.get_ids()
    
    print(f"Number of samples: {len(contexts)}")
    print("Running inference...")
    
    responses = run_inference_fn(contexts, instructions)
    
    print("Computing metrics...")
    scores = []
    for _id, response in zip(ids, responses):
        score = handler.compute_metric(_id, response)
        scores.append(score)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    result = {
        "dataset": dataset_name,
        "num_samples": len(contexts),
        "average_score": avg_score,
        "individual_scores": scores
    }
    
    print(f"Average score: {avg_score:.4f}")
    print()
    
    return result

def run_all_benchmarks(run_inference_fn: Callable) -> Dict[str, Any]:
    available_datasets = [
        "hotpotqa",
    ]
    
    results = {}
    
    for dataset_name in available_datasets:
        result = run_benchmark(dataset_name, run_inference_fn)
        results[dataset_name] = result
    
    if results:
        overall_avg = sum(r["average_score"] for r in results.values()) / len(results)
        results["overall_average"] = overall_avg
        
        print(f"{'='*80}")
        print(f"Overall average score: {overall_avg:.4f}")
        print(f"{'='*80}")
    
    return results