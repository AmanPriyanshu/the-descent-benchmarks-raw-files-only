# utils.py
import re
from collections import Counter, namedtuple

Score = namedtuple("Score", ["precision", "recall", "fmeasure"])

class RougeScorer:
    def __init__(self, rouge_types=None, use_stemmer=False):
        self.rouge_types = rouge_types or ['rougeL']
        self.use_stemmer = use_stemmer
    
    def score(self, target, prediction):
        target_tokens = self._tokenize(target)
        prediction_tokens = self._tokenize(prediction)
        
        result = {}
        for rouge_type in self.rouge_types:
            if rouge_type == 'rougeL':
                result[rouge_type] = self._score_lcs(target_tokens, prediction_tokens)
            elif rouge_type.startswith('rouge') and rouge_type[5:].isdigit():
                n = int(rouge_type[5:])
                target_ngrams = self._create_ngrams(target_tokens, n)
                prediction_ngrams = self._create_ngrams(prediction_tokens, n)
                result[rouge_type] = self._score_ngrams(target_ngrams, prediction_ngrams)
        
        return result
    
    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9]+', ' ', text)
        tokens = text.split()
        return [t for t in tokens if t]
    
    def _create_ngrams(self, tokens, n):
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] += 1
        return ngrams
    
    def _score_ngrams(self, target_ngrams, prediction_ngrams):
        intersection_count = 0
        for ngram in target_ngrams:
            intersection_count += min(target_ngrams[ngram], prediction_ngrams[ngram])
        
        target_count = sum(target_ngrams.values())
        prediction_count = sum(prediction_ngrams.values())
        
        precision = intersection_count / max(prediction_count, 1)
        recall = intersection_count / max(target_count, 1)
        fmeasure = self._compute_fmeasure(precision, recall)
        
        return Score(precision=precision, recall=recall, fmeasure=fmeasure)
    
    def _score_lcs(self, target_tokens, prediction_tokens):
        if not target_tokens or not prediction_tokens:
            return Score(precision=0, recall=0, fmeasure=0)
        
        lcs_length = self._lcs_length(target_tokens, prediction_tokens)
        
        precision = lcs_length / len(prediction_tokens)
        recall = lcs_length / len(target_tokens)
        fmeasure = self._compute_fmeasure(precision, recall)
        
        return Score(precision=precision, recall=recall, fmeasure=fmeasure)
    
    def _lcs_length(self, x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def _compute_fmeasure(self, precision, recall):
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0