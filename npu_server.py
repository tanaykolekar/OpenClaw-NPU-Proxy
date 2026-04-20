import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer
from ipex_llm.transformers.npu_model import AutoModelForCausalLM

# --- 1. INITIALIZE HARDWARE & MODEL ---
os.environ["IPEX_LLM_NPU_MTL"] = "1"
model_path = "C:\\Users\\tkole\\Massive_NPU_Model"

print("Booting custom NPU Server (Ollama Disguise Mode)...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.load_low_bit(model_path, trust_remote_code=True)
print("NPU Server is ONLINE and disguised as Ollama on Port 11434.")

app = FastAPI(title="NPU Ollama Proxy")

# --- 2. THE OLLAMA DIALECT DATA SHAPES ---
class Message(BaseModel):
    role: str
    content: str

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False

# --- 3. MOCKING THE MODEL LIST (So OpenClaw thinks the model is installed) ---
@app.get("/api/tags")
async def get_tags():
    return {"models": [{"name": "deepseek-npu:latest", "model": "deepseek-npu:latest"}]}

# --- 4. THE CHAT ENDPOINT ---
@app.post("/api/chat")
async def chat_completions(req: OllamaChatRequest):
    print(f"\n[OLLAMA DIALECT] Request from OpenClaw! Formatting {len(req.messages)} messages...")
    
    messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
    formatted_prompt = tokenizer.apply_chat_template(messages_dict, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    print("[PROCESSING] NPU is crunching the context...")
    
# Generate with "Strict Robotic" Guardrails
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=150,          # FORCE it to be concise (down from 800)
        temperature=0.1,             # Nearly 0 creativity. Makes it highly predictable.
        repetition_penalty=1.15,     # A balanced penalty so it can use normal words again
        top_k=10,                    # Only look at the top 10 most logical next words
        top_p=0.8,
        do_sample=True
    )
    
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"[SUCCESS] Sending response back to OpenClaw.")
    
    # Return exactly what Ollama would return
    return {
        "model": req.model,
        "created_at": "2026-04-16T12:00:00Z",
        "message": {
            "role": "assistant",
            "content": response_text
        },
        "done": True
    }

if __name__ == "__main__":
    # Ollama uses port 11434 by default!
    uvicorn.run(app, host="127.0.0.1", port=11434)