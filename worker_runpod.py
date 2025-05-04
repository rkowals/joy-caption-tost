# ...

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image_url = values.get('input_image_url')
    vlm_prompt = values.get('vlm_prompt', "")
    max_new_tokens = values.get('max_new_tokens', 64)
    top_k = values.get('top_k', 50)
    temperature = values.get('temperature', 1.0)

    input_image = download_file(input_image_url)
    input_image = Image.open(input_image).convert("RGB")

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
