from fastapi import FastAPI
from transformers import LlamaForCausalLM, LlamaTokenizer

app = FastAPI()

model_name = "Llama-3.2-1B-Instruct.Q6_K.llamafile"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

@app.get("/generate")
def generate(input_text: str):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
