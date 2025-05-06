import os, json, tempfile, requests, runpod
import io
import torch
import base64
from torch import nn
import torch.amp.autocast_mode
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM

def download_file(url, save_dir='/content/input'):
    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

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

CLIP_PATH = "/content/siglip"
MODEL_PATH = "/content/llama"

with torch.inference_mode():
    clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
    clip_model = AutoModel.from_pretrained(CLIP_PATH)
    clip_model = clip_model.vision_model
    clip_model.eval()
    clip_model.requires_grad_(False)
    clip_model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
    text_model.eval()
    image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size)
    image_adapter.load_state_dict(torch.load("/content/adapter/image_adapter.pt", map_location="cpu"))
    image_adapter.eval()
    image_adapter.to("cuda")

# ...

@torch.inference_mode()
def generate(input):
    values = input["input"]


    # 1) Decode Base64 if provided
    b64 = values.get("input_image_base64")
    if b64:
        input_image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    else:
        # 2) Fallback to URL
        url = values.get("input_image_url")
        if not url:
            return {"status": "FAILED", "error": "No image provided"}
        file_path = download_file(url)
        input_image = Image.open(file_path).convert("RGB")

    vlm_prompt = values.get('vlm_prompt', "")
    max_new_tokens = values.get('max_new_tokens', 64)
    top_k = values.get('top_k', 50)
    temperature = values.get('temperature', 1.0)

    image = clip_processor(images=input_image, return_tensors='pt').pixel_values
    image = image.to('cuda')
    prompt = tokenizer.encode(vlm_prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)
    
    with torch.amp.autocast_mode.autocast('cuda', enabled=True):
        vision_outputs = clip_model(pixel_values=image, output_hidden_states=True)
        image_features = vision_outputs.hidden_states[-2]
        embedded_images = image_adapter(image_features).to('cuda')

    prompt_embeds = text_model.model.embed_tokens(prompt.to('cuda'))
    embedded_bos = text_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64))

    inputs_embeds = torch.cat([
        embedded_bos.expand(embedded_images.shape[0], -1, -1),
        embedded_images.to(dtype=embedded_bos.dtype),
        prompt_embeds.expand(embedded_images.shape[0], -1, -1),
    ], dim=1)

    input_ids = torch.cat([
        torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
        torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
        prompt,
    ], dim=1).to('cuda')

    attention_mask = torch.ones_like(input_ids)
    generate_ids = text_model.generate(
        input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=top_k,
        temperature=temperature
    )
    generate_ids = generate_ids[:, input_ids.shape[1]:]
    if generate_ids[0][-1] == tokenizer.eos_token_id:
        generate_ids = generate_ids[:, :-1]
    caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        file_path = temp_file.name
        temp_file.write(caption.strip().encode('utf-8'))

    result = file_path

    # --- NOTYFIKACJA ---
    try:
        notify_uri = values.pop('notify_uri', None)
        notify_token = values.pop('notify_token', None)
        discord_id = values.pop('discord_id', os.getenv('com_camenduru_discord_id'))
        discord_channel = values.pop('discord_channel', os.getenv('com_camenduru_discord_channel'))
        discord_token = values.pop('discord_token', os.getenv('com_camenduru_discord_token'))
        job_id = values.pop('job_id', 'unknown_job')

        if all([discord_id, discord_channel, discord_token]):
            default_filename = os.path.basename(result)
            with open(result, "rb") as file:
                files = {default_filename: file.read()}
            payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
            response = requests.post(
                f"https://discord.com/api/v9/channels/{discord_channel}/messages",
                data=payload,
                headers={"Authorization": f"Bot {discord_token}"},
                files=files
            )
            response.raise_for_status()
            result_url = response.json()['attachments'][0]['url']
        else:
            result_url = None

        notify_payload = {"jobId": job_id, "result": result_url or "No Discord URL", "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')

        if web_notify_uri and web_notify_token:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        if notify_uri and notify_token:
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})

        return {"jobId": job_id, "result": result_url or caption, "status": "DONE"}

    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if web_notify_uri and web_notify_token:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            if notify_uri and notify_token:
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})
