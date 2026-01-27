# Long Context Understanding Benchmark: Raw Files Only

Standardized benchmark collection for The-Descent evaluation harness. Contains 36 long-context datasets from LongBench v1 (subset), LongBench v2, and OOLONG, totaling 8,925 evaluation instances with unified schema across multi-hop reasoning, summarization, code understanding, and aggregation tasks spanning 2,847-16,182,936 characters.

## Benchmark Composition

| Category | Number of Datasets |
|----------|-------------------|
| LongBench v1 | 12 |
| LongBench v2 | 20 |
| OOLONG | 4 |
| **Total** | **36** |

## Context Length Distribution

Distribution of 5,194 unique context documents across different length ranges:

| Context Length | Characters | GPT-2 Tokens | Llama-3.1-8B Tokens | GPT-OSS-20B Tokens |
|---------------|-----------|--------------|---------------------|-------------------|
| 0-8K | 68 | 1,286 | 1,322 | 1,338 |
| 8K-16K | 290 | 2,126 | 2,198 | 2,210 |
| 16K-32K | 814 | 588 | 528 | 504 |
| 32K-64K | 1,694 | 212 | 204 | 202 |
| 64K-128K | 1,184 | 202 | 262 | 262 |
| 128K-256K | 206 | 334 | 264 | 266 |
| 256K-512K | 234 | 242 | 248 | 248 |
| 512K-1M | 318 | 104 | 78 | 76 |
| 1M-2M | 226 | 64 | 72 | 70 |
| 2M-4M | 82 | 26 | 18 | 16 |
| 4M-16M | 78 | 10 | - | 2 |

### Summary Statistics

| Metric | Characters | GPT-2 Tokens | Llama-3.1-8B Tokens | GPT-OSS-20B Tokens |
|--------|-----------|--------------|---------------------|-------------------|
| Min | 2,847 | 997 | 913 | 887 |
| Q1 | 35,614 | 8,264 | 8,080 | 7,933 |
| Median | 59,978 | 13,944 | 13,788 | 13,572 |
| Q3 | 96,492 | 23,160 | 22,456 | 22,277 |
| Max | 16,182,936 | 8,284,502 | 4,144,373 | 4,264,266 |
| Mean | 311,910 | 101,808 | 82,418 | 81,068 |
| Std Dev | 928,449 | 392,158 | 259,653 | 258,216 |

## Dataset Structure

Each benchmark contains:
- `queries.json`: Evaluation instances with questions and ground truth answers
- `contexts/`: Directory containing context documents as `.txt` files

All datasets follow a unified schema with standardized field names for consistent evaluation.

## License
For detailed licensing information, attribution requirements, and citations for all datasets, please refer to [LICENSES.md](LICENSES.md).
