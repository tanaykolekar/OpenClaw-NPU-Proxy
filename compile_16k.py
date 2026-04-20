import os
import torch
from transformers import AutoTokenizer
from ipex_llm.transformers.npu_model import AutoModelForCausalLM

# 1. Force the Meteor Lake hardware routing
os.environ["IPEX_LLM_NPU_MTL"] = "1"

print("Step 1: Initializing 16K Grid Compilation...")
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
save_folder = "C:\\Users\\tkole\\Massive_NPU_Model"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("Step 2: Streaming 16K NPU Graph to Disk...")
# 2. Pass the save_directory upfront so it doesn't blow up your RAM
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_low_bit="sym_int4",   
    optimize_model=True,          
    max_context_len=16384,        
    max_prompt_len=10000,         
    trust_remote_code=True,
    save_directory=save_folder    # <-- The missing puzzle piece!
)

print("Step 3: Saving Tokenizer...")
tokenizer.save_pretrained(save_folder)

print("Success! 16K Graph Compiled.")