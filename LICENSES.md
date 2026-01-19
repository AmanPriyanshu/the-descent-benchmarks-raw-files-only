# Dataset Licenses

## Overview

This benchmark suite contains **36 datasets** across three major categories with a total of **~8,000 evaluation instances**.

---

## Dataset Inventory

| Superset Benchmark | Dataset | License | Task Type | Number of Samples |
|-------------------|---------|---------|-----------|-------------------|
| LongBench-v1 | 2wikimqa | Apache 2.0 | Multi-hop QA | 200 |
| LongBench-v1 | gov_report | CC-BY-4.0 | Summarization | 200 |
| LongBench-v1 | hotpotqa | Apache 2.0 | Multi-hop QA | 200 |
| LongBench-v1 | multifieldqa_en | Apache 2.0 | Multi-domain QA | 150 |
| LongBench-v1 | musique | CC-BY-4.0 | Multi-hop reasoning | 200 |
| LongBench-v1 | narrativeqa | Apache 2.0 | Story understanding | 200 |
| LongBench-v1 | passage_count | Apache 2.0 | Counting task | 200 |
| LongBench-v1 | passage_retrieval_en | Apache 2.0 | Passage retrieval | 200 |
| LongBench-v1 | qasper | CC-BY-4.0 | Scientific papers QA | 200 |
| LongBench-v1 | qmsum | MIT | Meeting summarization | 200 |
| LongBench-v1 | samsum | GPL-3.0 | Dialogue summarization | 200 |
| LongBench-v1 | triviaqa | Apache 2.0 | Trivia QA | 200 |
| LongBench-v2 | longbench_v2_code_repository_understanding_code_repo_qa | Apache 2.0 | Code Repository Understanding | 50 |
| LongBench-v2 | longbench_v2_long_dialogue_history_understanding_agent_history_qa | Apache 2.0 | Long-dialogue History Understanding | 20 |
| LongBench-v2 | longbench_v2_long_dialogue_history_understanding_dialogue_history_qa | Apache 2.0 | Long-dialogue History Understanding | 19 |
| LongBench-v2 | longbench_v2_long_in_context_learning_many_shot_learning | Apache 2.0 | Long In-context Learning | 21 |
| LongBench-v2 | longbench_v2_long_in_context_learning_new_language_translation | Apache 2.0 | Long In-context Learning | 20 |
| LongBench-v2 | longbench_v2_long_in_context_learning_user_guide_qa | Apache 2.0 | Long In-context Learning | 40 |
| LongBench-v2 | longbench_v2_long_structured_data_understanding_knowledge_graph_reasoning | Apache 2.0 | Long Structured Data Understanding | 15 |
| LongBench-v2 | longbench_v2_long_structured_data_understanding_table_qa | Apache 2.0 | Long Structured Data Understanding | 18 |
| LongBench-v2 | longbench_v2_multi_document_qa_academic | Apache 2.0 | Multi-Document QA | 50 |
| LongBench-v2 | longbench_v2_multi_document_qa_financial | Apache 2.0 | Multi-Document QA | 15 |
| LongBench-v2 | longbench_v2_multi_document_qa_governmental | Apache 2.0 | Multi-Document QA | 23 |
| LongBench-v2 | longbench_v2_multi_document_qa_legal | Apache 2.0 | Multi-Document QA | 14 |
| LongBench-v2 | longbench_v2_multi_document_qa_multi_news | Apache 2.0 | Multi-Document QA | 23 |
| LongBench-v2 | longbench_v2_single_document_qa_academic | Apache 2.0 | Single-Document QA | 44 |
| LongBench-v2 | longbench_v2_single_document_qa_detective | Apache 2.0 | Single-Document QA | 22 |
| LongBench-v2 | longbench_v2_single_document_qa_event_ordering | Apache 2.0 | Single-Document QA | 20 |
| LongBench-v2 | longbench_v2_single_document_qa_financial | Apache 2.0 | Single-Document QA | 22 |
| LongBench-v2 | longbench_v2_single_document_qa_governmental | Apache 2.0 | Single-Document QA | 18 |
| LongBench-v2 | longbench_v2_single_document_qa_legal | Apache 2.0 | Single-Document QA | 19 |
| LongBench-v2 | longbench_v2_single_document_qa_literary | Apache 2.0 | Single-Document QA | 30 |
| OOLONG | oolong_real_multidoc_rolls | MIT / CC-BY-SA-4.0 | Counting & Aggregation | 2630 |
| OOLONG | oolong_real_multidoc_spells | MIT / CC-BY-SA-4.0 | Counting & Aggregation | 3051 |
| OOLONG | oolong_real_singledoc_rolls | MIT / CC-BY-SA-4.0 | Counting & Aggregation | 177 |
| OOLONG | oolong_real_singledoc_spells | MIT / CC-BY-SA-4.0 | Counting & Aggregation | 214 |

---

## Citation Requirements

### LongBench v1 & v2:
```bibtex
@article{bai2024longbench,
  title={LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks},
  author={Bai, Yushi and Tu, Shangqing and Zhang, Jiajie and Peng, Hao and Wang, Xiaozhi and Lv, Xin and Cao, Shulin and Xu, Jiazheng and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
  journal={arXiv preprint arXiv:2412.15204},
  year={2024}
}
```

### Oolong:
```bibtex
@misc{bertsch2025oolong,
  title={Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities},
  author={Bertsch, Amanda and Pratapa, Adithya and Mitamura, Teruko and Neubig, Graham and Gormley, Matthew R.},
  year={2025},
  eprint={2511.02817},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@inproceedings{rameshkumar-bailey-2020-storytelling,
  title={Storytelling with Dialogue: A Critical Role Dungeons and Dragons Dataset},
  author={Rameshkumar, Revanth and Bailey, Peter},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020},
  pages={5121--5134}
}
```

---

## Legal Notice

This benchmark suite is provided for research and evaluation purposes. Users are responsible for complying with all applicable licenses when using these datasets. For commercial use, ensure proper attribution and license compliance.

**Last Updated:** January 2026