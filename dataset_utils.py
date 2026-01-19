# dataset_utils.py
import json
from pathlib import Path
from collections import Counter
import re
from typing import List, Dict, Any, Tuple
from utils import RougeScorer

class BaseDatasetHandler:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_dir = Path(dataset_name)
        self.contexts_dir = self.dataset_dir / "contexts"
        self.queries = self._load_queries()
        self.ground_truths = self._build_ground_truth_map()
    
    def _load_queries(self) -> List[Dict]:
        queries_file = self.dataset_dir / "queries.json"
        with open(queries_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_ground_truth_map(self) -> Dict[str, Dict]:
        gt_map = {}
        for query in self.queries:
            gt_map[query['_id']] = {
                'answers': query['answers'],
                'dataset': query['dataset'],
                'all_classes': query['all_classes']
            }
        return gt_map
    
    def get_contexts_and_instructions(self) -> Tuple[List[str], List[str]]:
        contexts = []
        instructions = []
        
        for query in self.queries:
            context_id = query['context_id']
            context_file = self.contexts_dir / f"{context_id}.txt"
            
            with open(context_file, 'r', encoding='utf-8') as f:
                context = f.read()
            
            contexts.append(context)
            
            instruction = query['input']
            if not instruction or instruction.strip() == "":
                instruction = self._get_default_instruction()
            
            instructions.append(instruction)
        
        return contexts, instructions
    
    def _get_default_instruction(self) -> str:
        return "Please answer based on the context provided."
    
    def get_ids(self) -> List[str]:
        return [query['_id'] for query in self.queries]
    
    def get_ground_truth(self, _id: str) -> Dict:
        return self.ground_truths[_id]
    
    def compute_metric(self, _id: str, response: str) -> float:
        raise NotImplementedError("Subclasses must implement compute_metric")
    
    def _normalize_answer(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = re.sub(r'[^a-z0-9\s]', ' ', s)
        s = ' '.join(s.split())
        return s

class HotpotQAHandler(BaseDatasetHandler):
    def __init__(self):
        super().__init__("hotpotqa")
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.ground_truths[_id]
        answers = gt_info['answers']
        return self._compute_f1(response, answers)
    
    def _compute_f1(self, prediction: str, ground_truths: List[str]) -> float:
        pred_tokens = self._normalize_answer(prediction).split()
        
        f1_scores = []
        for gt in ground_truths:
            gt_tokens = self._normalize_answer(gt).split()
            
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                f1_scores.append(0.0)
                continue
            
            precision = num_same / len(pred_tokens) if pred_tokens else 0
            recall = num_same / len(gt_tokens) if gt_tokens else 0
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)
        
        return max(f1_scores) if f1_scores else 0.0

class WikiMultihopQAHandler(BaseDatasetHandler):
    def __init__(self):
        super().__init__("2wikimqa")
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.ground_truths[_id]
        answers = gt_info['answers']
        return self._compute_f1(response, answers)
    
    def _compute_f1(self, prediction: str, ground_truths: List[str]) -> float:
        pred_tokens = self._normalize_answer(prediction).split()
        
        f1_scores = []
        for gt in ground_truths:
            gt_tokens = self._normalize_answer(gt).split()
            
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                f1_scores.append(0.0)
                continue
            
            precision = num_same / len(pred_tokens) if pred_tokens else 0
            recall = num_same / len(gt_tokens) if gt_tokens else 0
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)
        
        return max(f1_scores) if f1_scores else 0.0

class MuSiQueHandler(BaseDatasetHandler):
    def __init__(self):
        super().__init__("musique")
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.ground_truths[_id]
        answers = gt_info['answers']
        return self._compute_f1(response, answers)
    
    def _compute_f1(self, prediction: str, ground_truths: List[str]) -> float:
        pred_tokens = self._normalize_answer(prediction).split()
        
        f1_scores = []
        for gt in ground_truths:
            gt_tokens = self._normalize_answer(gt).split()
            
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                f1_scores.append(0.0)
                continue
            
            precision = num_same / len(pred_tokens) if pred_tokens else 0
            recall = num_same / len(gt_tokens) if gt_tokens else 0
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)
        
        return max(f1_scores) if f1_scores else 0.0

class NarrativeQAHandler(BaseDatasetHandler):
    def __init__(self):
        super().__init__("narrativeqa")
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.ground_truths[_id]
        answers = gt_info['answers']
        return self._compute_f1(response, answers)
    
    def _compute_f1(self, prediction: str, ground_truths: List[str]) -> float:
        pred_tokens = self._normalize_answer(prediction).split()
        
        f1_scores = []
        for gt in ground_truths:
            gt_tokens = self._normalize_answer(gt).split()
            
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                f1_scores.append(0.0)
                continue
            
            precision = num_same / len(pred_tokens) if pred_tokens else 0
            recall = num_same / len(gt_tokens) if gt_tokens else 0
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)
        
        return max(f1_scores) if f1_scores else 0.0

class QasperHandler(BaseDatasetHandler):
    def __init__(self):
        super().__init__("qasper")
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.ground_truths[_id]
        answers = gt_info['answers']
        return self._compute_f1(response, answers)
    
    def _compute_f1(self, prediction: str, ground_truths: List[str]) -> float:
        pred_tokens = self._normalize_answer(prediction).split()
        
        f1_scores = []
        for gt in ground_truths:
            gt_tokens = self._normalize_answer(gt).split()
            
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                f1_scores.append(0.0)
                continue
            
            precision = num_same / len(pred_tokens) if pred_tokens else 0
            recall = num_same / len(gt_tokens) if gt_tokens else 0
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)
        
        return max(f1_scores) if f1_scores else 0.0

class GovReportHandler(BaseDatasetHandler):
    def __init__(self):
        super().__init__("gov_report")
    
    def _get_default_instruction(self) -> str:
        return "Please provide a comprehensive summary of this government report."
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.ground_truths[_id]
        answers = gt_info['answers']
        return self._compute_rouge_l(response, answers)
    
    def _compute_rouge_l(self, prediction: str, ground_truths: List[str]) -> float:
        scorer = RougeScorer(['rougeL'])
        
        rouge_scores = []
        for gt in ground_truths:
            scores = scorer.score(gt, prediction)
            rouge_scores.append(scores['rougeL'].fmeasure)
        
        return max(rouge_scores) if rouge_scores else 0.0

class QMSumHandler(BaseDatasetHandler):
    def __init__(self):
        super().__init__("qmsum")
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.ground_truths[_id]
        answers = gt_info['answers']
        return self._compute_rouge_l(response, answers)
    
    def _compute_rouge_l(self, prediction: str, ground_truths: List[str]) -> float:
        scorer = RougeScorer(['rougeL'])
        
        rouge_scores = []
        for gt in ground_truths:
            scores = scorer.score(gt, prediction)
            rouge_scores.append(scores['rougeL'].fmeasure)
        
        return max(rouge_scores) if rouge_scores else 0.0

class TriviaQAHandler(BaseDatasetHandler):
    """
    TriviaQA handler with specialized answer selection.
    
    TriviaQA contains answer variations in multiple languages and special characters.
    Examples that require special handling:
    - _id b74a2400b06d8426b297e757dc5e91c9e981b40e54251577: Arabic text 'الجمهورية العربية السورية'
    - _id bc12fa16754f05a520ff0077455060c1bbe531c96711aac8: Special symbol '℡'
    
    These normalize to empty strings with standard alphanumeric normalization.
    We select the first answer variant that produces valid tokens after normalization.
    """
    
    def __init__(self):
        super().__init__("triviaqa")
    
    def get_ground_truth(self, _id: str) -> Dict:
        """Override to return best answer variant."""
        gt = self.ground_truths[_id]
        # Find first answer that normalizes to non-empty
        best_answer = None
        for answer in gt['answers']:
            normalized = self._normalize_answer(answer)
            if normalized and normalized.split():
                best_answer = answer
                break
        
        if best_answer is None:
            best_answer = gt['answers'][0]
        
        # Return modified ground truth with best answer first
        return {
            'answers': [best_answer] + [a for a in gt['answers'] if a != best_answer],
            'dataset': gt['dataset'],
            'all_classes': gt['all_classes']
        }
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.get_ground_truth(_id)
        answers = gt_info['answers']
        return self._compute_f1(response, answers)
    
    def _compute_f1(self, prediction: str, ground_truths: List[str]) -> float:
        pred_tokens = self._normalize_answer(prediction).split()
        
        f1_scores = []
        for gt in ground_truths:
            gt_tokens = self._normalize_answer(gt).split()
            
            if not gt_tokens or not pred_tokens:
                f1_scores.append(0.0)
                continue
            
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                f1_scores.append(0.0)
                continue
            
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)
        
        return max(f1_scores) if f1_scores else 0.0

class SAMSumHandler(BaseDatasetHandler):
    def __init__(self):
        super().__init__("samsum")
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.ground_truths[_id]
        answers = gt_info['answers']
        return self._compute_rouge_l(response, answers)
    
    def _compute_rouge_l(self, prediction: str, ground_truths: List[str]) -> float:
        scorer = RougeScorer(['rougeL'])
        
        rouge_scores = []
        for gt in ground_truths:
            scores = scorer.score(gt, prediction)
            rouge_scores.append(scores['rougeL'].fmeasure)
        
        return max(rouge_scores) if rouge_scores else 0.0

class MultiFieldQAEnHandler(BaseDatasetHandler):
    def __init__(self):
        super().__init__("multifieldqa_en")
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.ground_truths[_id]
        answers = gt_info['answers']
        return self._compute_f1(response, answers)
    
    def _compute_f1(self, prediction: str, ground_truths: List[str]) -> float:
        pred_tokens = self._normalize_answer(prediction).split()
        
        f1_scores = []
        for gt in ground_truths:
            gt_tokens = self._normalize_answer(gt).split()
            
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                f1_scores.append(0.0)
                continue
            
            precision = num_same / len(pred_tokens) if pred_tokens else 0
            recall = num_same / len(gt_tokens) if gt_tokens else 0
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)
        
        return max(f1_scores) if f1_scores else 0.0

class PassageRetrievalEnHandler(BaseDatasetHandler):
    def __init__(self):
        super().__init__("passage_retrieval_en")
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.ground_truths[_id]
        answers = gt_info['answers']
        return self._compute_accuracy(response, answers)
    
    def _compute_accuracy(self, prediction: str, ground_truths: List[str]) -> float:
        pred_norm = self._normalize_answer(prediction)
        
        for gt in ground_truths:
            gt_norm = self._normalize_answer(gt)
            if pred_norm == gt_norm:
                return 1.0
        
        return 0.0

class PassageCountHandler(BaseDatasetHandler):
    def __init__(self):
        super().__init__("passage_count")
    
    def _get_default_instruction(self) -> str:
        return "How many unique paragraphs are in the given text?"
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.ground_truths[_id]
        answers = gt_info['answers']
        return self._compute_accuracy(response, answers)
    
    def _compute_accuracy(self, prediction: str, ground_truths: List[str]) -> float:
        pred_norm = self._normalize_answer(prediction)
        
        for gt in ground_truths:
            gt_norm = self._normalize_answer(gt)
            if pred_norm == gt_norm:
                return 1.0
        
        return 0.0

class LongBenchV2BaseHandler(BaseDatasetHandler):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
    
    def get_contexts_and_instructions(self) -> Tuple[List[str], List[str]]:
        contexts = []
        instructions = []
        
        for query in self.queries:
            context_id = query['context_id']
            context_file = self.contexts_dir / f"{context_id}.txt"
            
            with open(context_file, 'r', encoding='utf-8') as f:
                context = f.read()
            
            contexts.append(context)
            
            question = query['input']
            choices = f"A: {query['choice_A']}\nB: {query['choice_B']}\nC: {query['choice_C']}\nD: {query['choice_D']}"
            instruction = f"{question}\n\n{choices}\n\nAnswer with only the letter (A, B, C, or D)."
            
            instructions.append(instruction)
        
        return contexts, instructions
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.ground_truths[_id]
        answers = gt_info['answers']
        return self._compute_accuracy(response, answers)
    
    def _compute_accuracy(self, prediction: str, ground_truths: List[str]) -> float:
        pred_norm = self._normalize_answer(prediction)
        
        for gt in ground_truths:
            gt_norm = self._normalize_answer(gt)
            if pred_norm == gt_norm or pred_norm.startswith(gt_norm):
                return 1.0
        
        return 0.0

class LongBenchV2LongIncontextLearningNewLanguageTranslationHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_long_in_context_learning_new_language_translation")

class LongBenchV2SingleDocumentQAFinancialHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_single_document_qa_financial")

class LongBenchV2MultiDocumentQAGovernmentalHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_multi_document_qa_governmental")

class LongBenchV2SingleDocumentQAEventOrderingHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_single_document_qa_event_ordering")

class LongBenchV2SingleDocumentQAAcademicHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_single_document_qa_academic")

class LongBenchV2SingleDocumentQADetectiveHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_single_document_qa_detective")

class LongBenchV2LongDialogueHistoryUnderstandingAgentHistoryQAHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_long_dialogue_history_understanding_agent_history_qa")

class LongBenchV2CodeRepositoryUnderstandingCodeRepoQAHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_code_repository_understanding_code_repo_qa")

class LongBenchV2MultiDocumentQAAcademicHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_multi_document_qa_academic")

class LongBenchV2SingleDocumentQALiteraryHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_single_document_qa_literary")

class LongBenchV2LongIncontextLearningManyShotLearningHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_long_in_context_learning_many_shot_learning")

class LongBenchV2LongIncontextLearningUserGuideQAHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_long_in_context_learning_user_guide_qa")

class LongBenchV2MultiDocumentQAFinancialHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_multi_document_qa_financial")

class LongBenchV2LongStructuredDataUnderstandingTableQAHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_long_structured_data_understanding_table_qa")

class LongBenchV2SingleDocumentQAGovernmentalHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_single_document_qa_governmental")

class LongBenchV2MultiDocumentQAMultiNewsHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_multi_document_qa_multi_news")

class LongBenchV2LongStructuredDataUnderstandingKnowledgeGraphReasoningHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_long_structured_data_understanding_knowledge_graph_reasoning")

class LongBenchV2SingleDocumentQALegalHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_single_document_qa_legal")

class LongBenchV2LongDialogueHistoryUnderstandingDialogueHistoryQAHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_long_dialogue_history_understanding_dialogue_history_qa")

class LongBenchV2MultiDocumentQALegalHandler(LongBenchV2BaseHandler):
    def __init__(self):
        super().__init__("longbench_v2_multi_document_qa_legal")

class OolongRealBaseHandler(BaseDatasetHandler):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
    
    def get_contexts_and_instructions(self) -> Tuple[List[str], List[str]]:
        contexts = []
        instructions = []
        
        for query in self.queries:
            context_id = query['context_id']
            context_file = self.contexts_dir / f"{context_id}.txt"
            
            with open(context_file, 'r', encoding='utf-8') as f:
                context = f.read()
            
            contexts.append(context)
            
            instruction = f"{query['input']}\n\nReturn your final answer in \\boxed{{answer}} format."
            instructions.append(instruction)
        
        return contexts, instructions
    
    def compute_metric(self, _id: str, response: str) -> float:
        gt_info = self.ground_truths[_id]
        gold = self._parse_answer(gt_info['answers'][0])
        trimmed_output = self._parse_response(response)
        
        if isinstance(gold, int) and isinstance(trimmed_output, int):
            return 0.75 ** abs(gold - trimmed_output)
        
        elif isinstance(gold, str) and isinstance(trimmed_output, str):
            return float(gold.strip().lower() == trimmed_output.strip().lower())
        
        elif isinstance(gold, list) and isinstance(trimmed_output, list):
            # FIXED: Use multiset intersection to preserve duplicates
            from collections import Counter
            gold_counter = Counter(gold)
            pred_counter = Counter(trimmed_output)
            
            # Intersection of multisets
            overlap = sum((gold_counter & pred_counter).values())
            
            # Score as intersection / gold_count
            return overlap / len(gold) if gold else 0.0
        
        return 0.0
    
    def _parse_response(self, answer: str):
        match = re.search(r'\\boxed\{\\text\{([^}]*)\}\}', answer) or \
                re.search(r'\\boxed[\{]+([^}]*)[\}]+', answer)
        if match:
            answer = match.group(1)
        
        return self._parse_answer(answer)
    
    def _parse_answer(self, answer):
        try:
            return int(answer)
        except ValueError:
            pass
        
        if ',' in answer:
            return [item.strip() for item in answer.split(',') if item.strip()]
        
        return answer

class OolongRealSingledocRollsHandler(OolongRealBaseHandler):
    def __init__(self):
        super().__init__("oolong_real_singledoc_rolls")

class OolongRealSingledocSpellsHandler(OolongRealBaseHandler):
    def __init__(self):
        super().__init__("oolong_real_singledoc_spells")

class OolongRealMultidocRollsHandler(OolongRealBaseHandler):
    def __init__(self):
        super().__init__("oolong_real_multidoc_rolls")

class OolongRealMultidocSpellsHandler(OolongRealBaseHandler):
    def __init__(self):
        super().__init__("oolong_real_multidoc_spells")

def get_dataset_handler(dataset_name: str) -> BaseDatasetHandler:
    handlers = {
        "hotpotqa": HotpotQAHandler,
        "2wikimqa": WikiMultihopQAHandler,
        "musique": MuSiQueHandler,
        "narrativeqa": NarrativeQAHandler,
        "qasper": QasperHandler,
        "gov_report": GovReportHandler,
        "qmsum": QMSumHandler,
        "triviaqa": TriviaQAHandler,
        "samsum": SAMSumHandler,
        "multifieldqa_en": MultiFieldQAEnHandler,
        "passage_retrieval_en": PassageRetrievalEnHandler,
        "passage_count": PassageCountHandler,
        "longbench_v2_long_in_context_learning_new_language_translation": LongBenchV2LongIncontextLearningNewLanguageTranslationHandler,
        "longbench_v2_single_document_qa_financial": LongBenchV2SingleDocumentQAFinancialHandler,
        "longbench_v2_multi_document_qa_governmental": LongBenchV2MultiDocumentQAGovernmentalHandler,
        "longbench_v2_single_document_qa_event_ordering": LongBenchV2SingleDocumentQAEventOrderingHandler,
        "longbench_v2_single_document_qa_academic": LongBenchV2SingleDocumentQAAcademicHandler,
        "longbench_v2_single_document_qa_detective": LongBenchV2SingleDocumentQADetectiveHandler,
        "longbench_v2_long_dialogue_history_understanding_agent_history_qa": LongBenchV2LongDialogueHistoryUnderstandingAgentHistoryQAHandler,
        "longbench_v2_code_repository_understanding_code_repo_qa": LongBenchV2CodeRepositoryUnderstandingCodeRepoQAHandler,
        "longbench_v2_multi_document_qa_academic": LongBenchV2MultiDocumentQAAcademicHandler,
        "longbench_v2_single_document_qa_literary": LongBenchV2SingleDocumentQALiteraryHandler,
        "longbench_v2_long_in_context_learning_many_shot_learning": LongBenchV2LongIncontextLearningManyShotLearningHandler,
        "longbench_v2_long_in_context_learning_user_guide_qa": LongBenchV2LongIncontextLearningUserGuideQAHandler,
        "longbench_v2_multi_document_qa_financial": LongBenchV2MultiDocumentQAFinancialHandler,
        "longbench_v2_long_structured_data_understanding_table_qa": LongBenchV2LongStructuredDataUnderstandingTableQAHandler,
        "longbench_v2_single_document_qa_governmental": LongBenchV2SingleDocumentQAGovernmentalHandler,
        "longbench_v2_multi_document_qa_multi_news": LongBenchV2MultiDocumentQAMultiNewsHandler,
        "longbench_v2_long_structured_data_understanding_knowledge_graph_reasoning": LongBenchV2LongStructuredDataUnderstandingKnowledgeGraphReasoningHandler,
        "longbench_v2_single_document_qa_legal": LongBenchV2SingleDocumentQALegalHandler,
        "longbench_v2_long_dialogue_history_understanding_dialogue_history_qa": LongBenchV2LongDialogueHistoryUnderstandingDialogueHistoryQAHandler,
        "longbench_v2_multi_document_qa_legal": LongBenchV2MultiDocumentQALegalHandler,
        "oolong_real_singledoc_rolls": OolongRealSingledocRollsHandler,
        "oolong_real_singledoc_spells": OolongRealSingledocSpellsHandler,
        "oolong_real_multidoc_rolls": OolongRealMultidocRollsHandler,
        "oolong_real_multidoc_spells": OolongRealMultidocSpellsHandler,
    }
    
    if dataset_name not in handlers:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return handlers[dataset_name]()