import os
import torch
from transformers import AutoTokenizer
from ipex_llm.transformers.npu_model import AutoModelForCausalLM

# 1. Route the math to the NPU
os.environ["IPEX_LLM_NPU_MTL"] = "1"
model_path = "C:\\Users\\tkole\\Massive_NPU_Model"

print("Booting up custom 16K NPU Brain (Loading from SSD)...")

# 2. Load the custom compiled model directly from your hard drive
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.load_low_bit(model_path, trust_remote_code=True)

print("\n==================================================")
print("🧠 16K LOCAL NPU CHAT ONLINE (Type 'exit' to quit) 🧠")
print("==================================================\n")

# 3. The Chat Loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', '/bye']:
        print("Shutting down NPU...")
        break

    # Format the prompt for DeepSeek architecture
    messages = [{"role": "user", "content": user_input}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    print("NPU Processing...")
    
    # Generate the answer (WITH ANTI-LOOPING CONSTRAINTS!)
    outputs = model.generate(
        inputs.input_ids, 
        max_new_tokens=2000,         # You have plenty of room for long answers now!
        temperature=0.7,             # Makes it sound natural
        repetition_penalty=1.1,      # Stops the "But wait!" infinite loops
        do_sample=True               # Unlocks creative thinking
    )
    
    # Slice the output to only show the new answer
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"\nTan-AI: {response}\n")
    print("-" * 50)