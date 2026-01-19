# test.py
import json
from typing import List
from main import run_benchmark
from dataset_utils import get_dataset_handler
import random

def run_inference_perfect(contexts: List[str], instructions: List[str]) -> List[str]:
    return test_responses['perfect']

def run_inference_mutated(contexts: List[str], instructions: List[str]) -> List[str]:
    return test_responses['mutated']

def run_inference_broken(contexts: List[str], instructions: List[str]) -> List[str]:
    return test_responses['broken']

def prepare_test_responses(dataset_name: str):
    global test_responses
    
    handler = get_dataset_handler(dataset_name)
    ids = handler.get_ids()
    
    perfect = []
    mutated = []
    broken = []
    
    for idx, _id in enumerate(ids):
        ground_truth = handler.get_ground_truth(_id)
        answer = ground_truth['answers'][0]
        
        perfect.append(answer)
        
        if dataset_name.startswith('oolong_real_') or \
           dataset_name in ['passage_retrieval_en', 'passage_count']:
            mutated_answer = answer
        elif dataset_name.startswith('longbench_v2_'):
            mutated_answer = 'A'
        else:
            words = answer.split()
            if len(words) > 3:
                mutated_answer = random.choice(["I think", "Perhaps", "Maybe"]) + " " + answer
            else:
                mutated_answer = answer + " probably"
        mutated.append(mutated_answer)
        
        broken.append(random.choice([
            "I don't know",
            "The answer is 42",
            "XYZ123",
            "Error",
            "Cannot process"
        ]))
    
    test_responses = {
        'perfect': perfect,
        'mutated': mutated,
        'broken': broken
    }

test_responses = {}

datasets_to_test = [
    "hotpotqa", 
    "2wikimqa", 
    "musique", 
    "narrativeqa", 
    "qasper", 
    "gov_report", 
    "qmsum", 
    "triviaqa", 
    "samsum",
    "multifieldqa_en",
    "passage_retrieval_en",
    "passage_count",
    "longbench_v2_long_in_context_learning_new_language_translation",
    "longbench_v2_single_document_qa_financial",
    "longbench_v2_multi_document_qa_governmental",
    "longbench_v2_single_document_qa_event_ordering",
    "longbench_v2_single_document_qa_academic",
    "longbench_v2_single_document_qa_detective",
    "longbench_v2_long_dialogue_history_understanding_agent_history_qa",
    "longbench_v2_code_repository_understanding_code_repo_qa",
    "longbench_v2_multi_document_qa_academic",
    "longbench_v2_single_document_qa_literary",
    "longbench_v2_long_in_context_learning_many_shot_learning",
    "longbench_v2_long_in_context_learning_user_guide_qa",
    "longbench_v2_multi_document_qa_financial",
    "longbench_v2_long_structured_data_understanding_table_qa",
    "longbench_v2_single_document_qa_governmental",
    "longbench_v2_multi_document_qa_multi_news",
    "longbench_v2_long_structured_data_understanding_knowledge_graph_reasoning",
    "longbench_v2_single_document_qa_legal",
    "longbench_v2_long_dialogue_history_understanding_dialogue_history_qa",
    "longbench_v2_multi_document_qa_legal",
    "oolong_real_singledoc_rolls",
    "oolong_real_singledoc_spells",
    "oolong_real_multidoc_rolls",
    "oolong_real_multidoc_spells",
]

all_results = []

for dataset_name in datasets_to_test:
    print(f"\n{'='*80}")
    print(f"Testing {dataset_name.upper()} benchmark")
    print(f"{'='*80}\n")
    
    prepare_test_responses(dataset_name)
    
    result_perfect = run_benchmark(dataset_name, run_inference_perfect)
    result_mutated = run_benchmark(dataset_name, run_inference_mutated)
    result_broken = run_benchmark(dataset_name, run_inference_broken)
    
    all_results.append({
        'dataset': dataset_name,
        'perfect': result_perfect['average_score'],
        'mutated': result_mutated['average_score'],
        'broken': result_broken['average_score']
    })

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"{'Dataset':<70} {'Perfect':<15} {'Mutated':<15} {'Broken':<15}")
print("-"*80)
for result in all_results:
    print(f"{result['dataset']:<70} {result['perfect']:<15.4f} {result['mutated']:<15.4f} {result['broken']:<15.4f}")
print("="*80)