from google import genai
import time
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
import config
import os
from audio_to_text import extract_audio, transcribe_audio
import json
from prompts import prompts

def generate_actions(video_path, transcription, clip_index):
    load_dotenv()
    api_key = os.getenv("API_KEY")
    client = genai.Client(api_key=api_key)
    myfile = client.files.upload(file=video_path)

    prompt_number = "prompt10"
    prompt = prompts[prompt_number]["video_clip_prompt"]

    while myfile.state.name == "PROCESSING":
        time.sleep(5)
        myfile = client.files.get(name=myfile.name)

    if myfile.state.name == "FAILED":
        raise ValueError(f"File processing failed: {myfile.state.name}")

    response = client.models.generate_content(
        model=config.MODEL,
        contents=[myfile, prompt]
    )
    return response.text

def process_clips(clips, save_dir="clip_actions"):
    os.makedirs(save_dir, exist_ok=True)
    all_actions = []

    for idx, clip_path in enumerate(clips):
        print(f"Processing clip {idx}: {clip_path}")

        audio = extract_audio(clip_path, "audio_clips")
        transcription = transcribe_audio(audio)

        actions = generate_actions(clip_path, transcription, idx)

        try:
            actions_json = json.loads(actions)
        except json.JSONDecodeError:
            print(f"Invalid JSON for clip {idx}")
            actions_json = {
                "clip_index": idx,
                "actions": actions
            }

        all_actions.append(actions_json)

        save_path = os.path.join(save_dir, f"clip_{idx}_actions.json")
        with open(save_path, "w") as f:
            json.dump(actions_json, f, indent=2)

    return all_actions

