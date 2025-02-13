The files are located as follows:
$ unzip S4705454.zip
Archive: S4705454.zip
creating: S4705454/
inflating: S4705454/report.pdf
inflating: S4705454/stage1.sh
inflating: S4705454/requirements.txt
inflating: S4705454/rankerA.py
inflating: S4705454/rankerB.py
inflating: S4705454/rerankerA.py
inflating: S4705454/rerankerB.py
inflating: S4705454/rerankerC.py
inflating: S4705454/rerankerD.py
inflating: S4705454/ragStage.pdf
inflating: S4705454/RAGA.py
inflating: S4705454/RAGB.py
inflating: S4705454/RAGC.py
inflating: S4705454/RAGD.py
inflating: S4705454/MyRAGEval.ipynb
inflating: S4705454/MyRetEval.ipynb
inflating: S4705454/MyREADME.md
inflating: S4705454/output/rankerA.json
inflating: S4705454/output/rankerB.json
inflating: S4705454/output/rerankerA.json
inflating: S4705454/output/rerankerB.json
inflating: S4705454/output/rerankerC.json
inflating: S4705454/output/rerankerD.json
inflating: S4705454/output/llama2_rankerB.json
inflating: S4705454/output/llama2_rerankerD.json
inflating: S4705454/output/llama3_rankerB.json
inflating: S4705454/output/llama3_rerankerD.json
inflating: S4705454/output/mistral_rankerB.json
inflating: S4705454/output/mistral_rerankerD.json
inflating: S4705454/output/zephyr_rankerB.json
inflating: S4705454/output/zephyr_rerankerD.json
inflating: S4705454/output/zephyr_rerankerD.json
inflating: S4705454/time/output/llama2_rankerB.json
inflating: S4705454/time/output/llama2_rerankerD.json
inflating: S4705454/time/output/llama3_rankerB.json
inflating: S4705454/time/output/llama3_rerankerD.json
inflating: S4705454/time/output/mistral_rankerB.json
inflating: S4705454/time/output/mistral_rerankerD.json
inflating: S4705454/time/output/zephyr_rankerB.json
inflating: S4705454/time/output/zephyr_rerankerD.json
inflating: S4705454/time/output/zephyr_rerankerD.json

The above files can be described as follows:
• MyREADME.md - This current file that containing all the directories, any instructions to run the code and, any changes made.
• report.pdf - The 5 page experimental report.
• stage1.sh - Used to stage the batch scripts for the Retireval Stage (stage 1) for rankers A and B, and rerankers A, B, C, and D.
• requirements.txt - Generated using pip freeze > requirements.txt to ensure that this project can be reproduced.
• ranker[A-B].py - The different retrieval baselines used.
• reranker[A-D].py - Any reranking baselines you used.
• ragStage.sh - Used to stage the batch scripts for the Generation Stage (stage 2) for RAG[A-D].py files. ragStage.sh was run two times, where the first iteration was to run the RAG files using rankerB and the second iteration was to run the RAG files using rerankerD. The output_file and input_stage_1 for each RAG files can be toggled to acheive this (see RAG[A-D].py main for more information)
• rag[A-D].py - The different LLM text generation baselines
• MyRAGEval.ipynb - The Jupyter Notebook file which evaluates the RAG models + BERT Scores, along with creating graphs and diagrams for the generation evaluation part of the report.
• MyRetEval.ipynb - The Jupyter Notebook file which evaluates the rankers and rerankers + RPrec, along with creating graphs and diagrams for the retreival evaluation part of the report.
• output/ranker[A-B].json - The different retrieval baseline results.
• output/reranker[A-D].json - Any reranking baseline results.
• output/[LLM][input_stage_1].json - The different LLM text generation results.
• time/output/[LLM][input_stage_1].json - The time taken per query for each LLM and input stage 1 combinations.
