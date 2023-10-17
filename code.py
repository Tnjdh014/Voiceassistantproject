import whisper
from transformers import AutoTokenizer, pipeline
import transformers
import torch

model = whisper.load_model("base")
transcription = model.transcribe('current.m4a')
text = transcription["text"]

#NLP model
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer)

response = text_generator(text, do_sample=True, max_length=100)[0]["generated_text"]
print(response)
