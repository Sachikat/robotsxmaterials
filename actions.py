from google import genai
import time
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
import config
import os
from audio_to_text import extract_audio, transcribe_audio
import json

def generate_actions(video_path, transcription, clip_index):
    load_dotenv()
    api_key = os.getenv("API_KEY")
    client = genai.Client(api_key=api_key)
    myfile = client.files.upload(file=video_path)

    prompt = f"""
        You are an expert observer in a material science laboratory.
        Your task is to analyze a short video clip and extract the lowest-level
        human actions being performed.

        You observe:
        - Physical actions performed by the human
        - Chemicals, tools, and containers used
        - Any measurements or quantities involved
        - Repeated or conditional steps

        Input:
        You are given:
        1. A video clip (8–10 seconds) of a human performing actions in a materials science lab.
        2. An optional audio transcription of the clip.
        - If the audio contains spoken instructions, use it to inform the actions.
        - If the audio is background noise, ignore it.

        Transcription of the video's audio:
        \"\"\"{transcription}\"\"\"

        Goal:
        Extract the **primitive, low-level actions** performed by the human.
        Use the video as the primary source of truth.
        Use audio only when it provides explicit procedural information.

        Guidelines:
        - List actions in the **exact temporal order** they occur.
        - Actions must be **atomic** (one physical action per step).
        - Include tools, chemicals, and measurements when visible or stated.
        - If an action is repeated consecutively, record it once and set "repeat" > 1.
        - If a step is performed **only under a condition**, mark it as an additional step
        and briefly describe the condition.
        - Do NOT infer intent or future steps beyond what is visible or stated.
        - If no meaningful lab actions are observable in the clip, return an empty "actions" list.
        - Do not hallucinate actions when none are visible or stated.

        Output:
        Return **ONLY valid JSON**.
        Do not include explanations, markdown, or commentary.

        Use this exact format:

        {{
        "clip_index": {clip_index},
        "actions": [
            {{
            "step": 1,
            "action": "pick up reagent bottle",
            "repeat": 1,
            "additional_step": false,
            "condition": null
            }},
            {{
            "step": 2,
            "action": "scoop powder (≈5 g)",
            "repeat": 1,
            "additional_step": false,
            "condition": null
            }},
            {{
            "step": 3,
            "action": "shake excess powder off scoop",
            "repeat": 1,
            "additional_step": true,
            "condition": "performed only if excess powder is visible"
            }}
        ]
        }}    
    """

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

