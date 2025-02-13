#!/bin/bash
# Set the CUDA device to 0 and run the Python script
# this script was run two times; one for ranker B and another one for reranker D which can be togelled in each of RAG[A-D].py files
export CUDA_VISIBLE_DEVICES=0

echo "Running meta-llama/Llama-2-7b-chat-hf"
START_TIME=$(date +%s)
python3 RAGA.py 
END_TIME=$(date +%s)
echo "Time taken: $(($END_TIME - $START_TIME)) seconds."

echo "Running meta-llama/Llama-3.1-8B-Instruct"
START_TIME=$(date +%s)
python3 RAGB.py 
END_TIME=$(date +%s)
echo "Time taken: $(($END_TIME - $START_TIME)) seconds."

echo "Running mistralai/Mistral-7B-Instruct-v0.1"
START_TIME=$(date +%s)
python3 RAGC.py 
END_TIME=$(date +%s)
echo "Time taken: $(($END_TIME - $START_TIME)) seconds."

echo "Running HuggingFaceH4/zephyr-7b-alpha"
START_TIME=$(date +%s)
python3 RAGD.py 
END_TIME=$(date +%s)
echo "Time taken: $(($END_TIME - $START_TIME)) seconds."

