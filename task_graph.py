import os
import json
from google import genai
import time
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
import config
import os
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type
from google.genai.errors import ServerError
from prompts import prompts


def clean_clip_json(output_file, input_dir="clip_actions"):
    """
    Fixes all JSON files in input_dir by converting actions strings into proper lists.
    Merges all clips into one JSON array.
    """
    output_file = f"{output_file}.txt"
    all_clips = []

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r") as f:
            data = json.load(f)

        # 🔑 Normalize: make everything a list of dicts
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            print(f"Skipping {filename}: unexpected JSON format")
            continue

        for clip in data:
            if not isinstance(clip, dict):
                continue

            actions = clip.get("actions")

            if isinstance(actions, str):
                actions_str = (
                    actions.replace("```json", "")
                           .replace("```", "")
                           .strip()
                )
                try:
                    parsed = json.loads(actions_str)
                    if isinstance(parsed, dict) and "actions" in parsed:
                        clip["actions"] = parsed["actions"]
                except json.JSONDecodeError:
                    print(f"Failed to parse actions in {filename}, keeping raw string")

            all_clips.append(clip)

    with open(output_file, "w") as f:
        json.dump(all_clips, f, indent=2)

    print(f"Cleaned JSON saved to {output_file}")
    return all_clips, output_file


def generate_task_graph(file_name, json_file):
    load_dotenv()
    api_key = os.getenv("API_KEY")

    client = genai.Client(api_key=api_key)

    myfile = client.files.upload(file=json_file)

    while myfile.state.name == "PROCESSING":
        time.sleep(30)
        myfile = client.files.get(name=myfile.name)

    if myfile.state.name == "FAILED":
        raise ValueError(f"File processing failed: {myfile.state.name}")
    
    prompt_number = "prompt10"
    prompt = prompts[prompt_number]["task_graph_prompt"]

    @retry(
        retry=retry_if_exception_type(ServerError),
        wait=wait_fixed(30),
        stop=stop_after_attempt(5)
    )
    def call_api():
        return client.models.generate_content(
            model=config.MODEL,
            contents=[myfile, prompt]
    )

    response = call_api()

    for part in response.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = part.as_image()
            image.save((file_name + ".png"))

    with open((file_name + ".txt"), "w") as f:
        f.write(response.text)