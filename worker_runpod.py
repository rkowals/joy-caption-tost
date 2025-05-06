import os
import json
tempfile
import requests
import base64
import io
import runpod

import torch
from torch import nn
import torch.amp.autocast_mode
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM

# --- Utility to download by URL ---
def download_file(url, save_dir='/content/input'):
    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

# --- Image Adapter Definition ---
class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)

    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x

# --- Paths and Model Loading ---
CLIP_PATH = "/content/siglip"
MODEL_PATH = "/content/llama"

with torch.inference_mode():
    # Load CLIP vision model
    clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
    clip_model = AutoModel.from_pretrained(CLIP_PATH).vision_model
    clip_model.eval().requires_grad_(False).to("cuda")

    # Load tokenizer and text model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    text_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16
    )
    text_model.eval()

    # Load image adapter
    image_adapter = ImageAdapter(
        clip_model.config.hidden_size,
        text_model.config.hidden_size
    )
    image_adapter.load_state_dict(
        torch.load("/content/adapter/image_adapter.pt", map_location="cpu")
    )
    image_adapter.eval().to("cuda")


@torch.inference_mode()
def generate(input):
    values = input.get("input", {})

    # 1) Decode Base64 if provided
    b64 = values.get("input_image_base64")
    if b64:
        image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    else:
        # 2) Fallback to URL
        url = values.get("input_image_url")
        if not url:
            return {"status": "FAILED", "error": "No image provided"}
        file_path = download_file(url)
        image = Image.open(file_path).convert("RGB")

    # Preprocess image
    pixel_values = clip_processor(images=image, return_tensors='pt').pixel_values.to('cuda')

    # Encode text prompt
    vlm_prompt = values.get('vlm_prompt', "")
    prompt_ids = tokenizer.encode(
        vlm_prompt,
        return_tensors='pt',
        padding=False,
        truncation=False,
        add_special_tokens=False
    )

    # Compute vision features and adapt
    with torch.amp.autocast('cuda', enabled=True):
        vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
        image_features = vision_outputs.hidden_states[-2]
        adapted = image_adapter(image_features).to('cuda')

    # Prepare combined embeddings
    prompt_embeds = text_model.model.embed_tokens(prompt_ids.to('cuda'))
    bos = tokenizer.bos_token_id
    embed_bos = text_model.model.embed_tokens(
        torch.tensor([[bos]], device=text_model.device)
    )

    inputs_embeds = torch.cat([
        embed_bos.expand(adapted.size(0), -1, -1),
        adapted,
        prompt_embeds.expand(adapted.size(0), -1, -1)
    ], dim=1)

    # Build dummy input_ids for compatibility
    input_ids = torch.full(
        (1, inputs_embeds.size(1)),
        tokenizer.pad_token_id or -100,
        dtype=torch.long,
        device='cuda'
    )
    input_ids[:, 0] = bos
    attention_mask = torch.ones_like(input_ids)

    # Generate text
    gen_args = {
        'inputs_embeds': inputs_embeds,
        'attention_mask': attention_mask,
        'max_new_tokens': values.get('max_new_tokens', 64),
        'do_sample': True,
        'top_k': values.get('top_k', 50),
        'temperature': values.get('temperature', 1.0)
    }
    output = text_model.generate(**gen_args)
    # Strip the conditioning tokens
    generated = output[:, inputs_embeds.size(1):]
    if generated[0, -1] == tokenizer.eos_token_id:
        generated = generated[:, :-1]
    caption = tokenizer.decode(
        generated[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Save result to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(caption.strip().encode('utf-8'))
        result_path = tmp.name

    # Notification logic (Discord, webhooks) remains unchanged...
    # [omitted for brevity]

    return {"status": "DONE", "jobId": values.get('job_id', ''), "result_path": result_path}

# Start RunPod server
runpod.serverless.start({"handler": generate})
