#!/bin/bash
# Set the CUDA device to 0 and run the Python script
export CUDA_VISIBLE_DEVICES=0

echo "Running BAAI/llm-embedder"
START_TIME=$(date +%s)
python3 rankerA.py "BAAI/llm-embedder" "output/rankerA.json"
END_TIME=$(date +%s)
echo "Time taken: $(($END_TIME - $START_TIME)) seconds."

echo "Running BAAI/llm-embedder reranked with BAAI/bge-reranker-base."
START_TIME=$(date +%s)
python3 rerankerA.py "BAAI/llm-embedder" "BAAI/bge-reranker-base" "output/rerankerA.json"
END_TIME=$(date +%s)
echo "Time taken: $(($END_TIME - $START_TIME)) seconds."

echo "Running BAAI/llm-embedder reranked with BAAI/bge-reranker-large."
START_TIME=$(date +%s)
python3 rerankerB.py "BAAI/llm-embedder" "BAAI/bge-reranker-large" "output/rerankerB.json"
END_TIME=$(date +%s)
echo "Time taken: $(($END_TIME - $START_TIME)) seconds."

echo "Running BAAI/bge-large-en-v1.5"
START_TIME=$(date +%s)
python3 rankerB.py "BAAI/bge-large-en-v1.5" "output/rankerB.json"
END_TIME=$(date +%s)
echo "Time taken: $(($END_TIME - $START_TIME)) seconds."

echo "Running BAAI/bge-large-en-v1.5 reranked with BAAI/bge-reranker-base."
START_TIME=$(date +%s)
python3 rerankerC.py "BAAI/bge-large-en-v1.5" "BAAI/bge-reranker-base" "output/rerankerC.json"
END_TIME=$(date +%s)
echo "Time taken: $(($END_TIME - $START_TIME)) seconds."

echo "Running BAAI/bge-large-en-v1.5 reranked with BAAI/bge-reranker-large."
START_TIME=$(date +%s)
python3 rerankerD.py "BAAI/bge-large-en-v1.5" "BAAI/bge-reranker-large" "output/rerankerD.json"
END_TIME=$(date +%s)
echo "Time taken: $(($END_TIME - $START_TIME)) seconds."

