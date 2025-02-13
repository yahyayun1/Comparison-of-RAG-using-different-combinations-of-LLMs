# Retrieval-Augmented Generation (RAG) Evaluation

## Overview
This repository contains the code and data for evaluating different retrieval and reranking baselines, as well as multiple RAG models using large language models (LLMs). The project consists of two main stages:
1. **Retrieval Stage** - Implementing and evaluating retrieval and reranking models.
2. **Generation Stage** - Using LLMs for text generation based on retrieved documents.

## Directory Structure
```
│-- report.pdf
│-- MyREADME.md
│-- stage1.sh
│-- requirements.txt
│-- rankerA.py
│-- rankerB.py
│-- rerankerA.py
│-- rerankerB.py
│-- rerankerC.py
│-- rerankerD.py
│-- ragStage.sh
│-- RAGA.py
│-- RAGB.py
│-- RAGC.py
│-- RAGD.py
│-- MyRAGEval.ipynb
│-- MyRetEval.ipynb
│-- output/
│   │-- rankerA.json
│   │-- rankerB.json
│   │-- rerankerA.json
│   │-- rerankerB.json
│   │-- rerankerC.json
│   │-- rerankerD.json
│   │-- llama2_rankerB.json
│   │-- llama2_rerankerD.json
│   │-- llama3_rankerB.json
│   │-- llama3_rerankerD.json
│   │-- mistral_rankerB.json
│   │-- mistral_rerankerD.json
│   │-- zephyr_rankerB.json
│   │-- zephyr_rerankerD.json
│-- time/output/
│   │-- llama2_rankerB.json
│   │-- llama2_rerankerD.json
│   │-- llama3_rankerB.json
│   │-- llama3_rerankerD.json
│   │-- mistral_rankerB.json
│   │-- mistral_rerankerD.json
│   │-- zephyr_rankerB.json
│   │-- zephyr_rerankerD.json
```

## File Descriptions
- **MyREADME.md** - This file, containing details about the project, directory structure, and instructions.
- **report.pdf** - A 5-page experimental report summarizing findings.
- **stage1.sh** - Batch script for running rankers (A, B) and rerankers (A, B, C, D).
- **requirements.txt** - Dependencies required to run the project (`pip install -r requirements.txt`).
- **ranker[A-B].py** - Different retrieval baseline models.
- **reranker[A-D].py** - Various reranking baseline models.
- **ragStage.sh** - Batch script for running RAG models.
- **rag[A-D].py** - Different LLM-based text generation models.
- **MyRAGEval.ipynb** - Jupyter Notebook for evaluating RAG models (BERT scores, generation evaluation, graphs, and diagrams).
- **MyRetEval.ipynb** - Jupyter Notebook for evaluating retrieval models (RPrec, ranking performance, graphs, and diagrams).
- **output/** - Contains JSON results for retrieval, reranking, and LLM text generation.
- **time/output/** - Contains JSON logs for time taken per query.

## Running the Project
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the retrieval stage:
   ```bash
   bash stage1.sh
   ```
3. Run the generation stage:
   ```bash
   bash ragStage.sh
   ```
4. Evaluate retrieval and generation performance:
   - Open `MyRetEval.ipynb` to analyze retrieval results.
   - Open `MyRAGEval.ipynb` to analyze generation results.

## Results and Evaluation
- **Retrieval Performance** is measured using metrics like **RPrec**.
- **Generation Performance** is evaluated using **BERT Scores** and other text quality metrics.
- Execution time per query is recorded in the `time/output/` directory.

## License
This project is for research and academic purposes.

---
For any questions or contributions, feel free to open an issue or submit a pull request.
