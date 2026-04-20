# OpenClaw NPU Proxy: The "Ollama Trojan Horse" 🦞🧠

This project provides a custom FastAPI proxy server that allows locked-down enterprise agent frameworks (like OpenClaw) to execute massive 10,000+ token context windows completely offline using Intel's experimental Meteor Lake NPU.

## The Problem
1. **Hardware Limits:** Intel's NPU wrapper natively crashes (C++ Segfault) when trying to map standard 16K memory grids.
2. **Software Limits:** Frameworks like OpenClaw strictly enforce API keys, hardcoded cloud URLs (api.openai.com), and minimum context window checks, actively blocking local NPU routing.

## The Solution
This repository bypasses both restrictions:
1. **`compile_16k.py`**: A custom script that mathematically reduces the NPU's prefill matrix from 15K to 10K, successfully compiling a 16,384-token context window locally without crashing the C++ driver.
2. **`npu_server.py`**: A lightweight FastAPI server that hosts the local NPU graph but disguises itself as an **Ollama** server (listening on port `11434` and speaking the `/api/chat` dialect). 

Because frameworks trust local Ollama instances natively, this "Trojan Horse" bypasses all OpenAI API key requirements and URL validations, funneling the massive agentic payload directly into the offline silicon.

## How to Use
1. Run `compile_16k.py` to compile your downloaded HuggingFace model into an OpenVINO XML graph on your SSD.
2. Run `npu_server.py` to start the proxy.
3. In your Agent Framework, set your provider/model to `ollama/deepseek-npu` and watch the hardware crunch!


Disclaimer: This project is an independent, open-source networking tool designed for educational purposes and local hardware optimization. It is not affiliated with, endorsed by, or associated with OpenClaw, OpenAI, Ollama, or Intel. All trademarks belong to their respective owners.