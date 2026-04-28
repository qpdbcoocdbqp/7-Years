# uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
# uv pip install -U "transformers>=4.36,<4.58.0"
import os
from pathlib import Path

checkpoint = Path(os.getenv("HOME")) /  ".cache/huggingface/hub/models--openai--privacy-filter/snapshots/7ffa9a043d54d1be65afb281eddf0ffbe629385b"
os.listdir(checkpoint)
# ['config.json', 'onnx', 'original', 'tokenizer.json', 'tokenizer_config.json', 'viterbi_calibration.json']
  
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
# Load quantized ONNX version
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = ORTModelForTokenClassification.from_pretrained(
    checkpoint, 
    file_name="model_q4.onnx"
)


#### offical example
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForTokenClassification.from_pretrained(checkpoint, device_map="auto")

inputs = tokenizer("My name is Alice Smith", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

predicted_token_class_ids = outputs.logits.argmax(dim=-1)
predicted_token_classes = [model.config.id2label[token_id.item()] for token_id in predicted_token_class_ids[0]]
print(predicted_token_classes)
