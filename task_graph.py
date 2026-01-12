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

        if isinstance(data.get("actions"), str):
            actions_str = data["actions"].replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(actions_str)
                if "actions" in parsed:
                    data["actions"] = parsed["actions"]
            except json.JSONDecodeError:
                print(f"Failed to parse actions in {filename}, keeping raw string")

        all_clips.append(data)

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
    
    prompt = """
        You are an expert in procedural task modeling.

        You are given a one text file formatted like a JSON file.
        - You are given an ordered list of atomic human actions split into clips.
        - Each action may include a conditional or repeat information.
        This text file is a list of JSON files, each representing a short video clip from a materials science lab. 
        Each clip contains:
        - "clip_index": the index of the clip
        - "actions": a list of atomic steps performed in that clip
        - Each action includes:
            - Step number within the clip
            - Action description
            - Repeat count
            - Whether it is conditional/optional

        Your task:
        1. Construct a clean **task graph** showing the entire procedural flow.
        - Use **standard nodes**, like A, B, C... for steps.
        - Arrows (→) indicate the temporal or conditional sequence.
        - Respect the **order of steps within each clip**, and then **across clips**.
        - Conditional branches should be shown as branching arrows, but **do not label nodes with clip numbers or step IDs**.
        - Do NOT invent steps.
        - Do NOT change the order of steps.
        2. Respect the temporal order of steps:
        - First, maintain the order of actions within each clip.
        - Then, place clips in increasing "clip_index" order.
        3. Represent repeated actions as loops when appropriate.
        4. Represent optional or conditional steps as branches.
        5. Do **not** invent any steps or change their order.

        Output:
        - Return **only a Mermaid flowchart** of the overall task graph.
        - Do **not** include explanations or markdown.xw    
        - Use conditional branching only when necessary, without extra labels.

        Example:

        graph TD
            A[Open cabinet door] --> B[Reach into cabinet]
            B --> C[Retrieve powder bottle]
            C --> D[Close cabinet door]
            D --> E[Hold powder bottle]
            E --> F[Move bottle to tray]
            F --> G[Place bottle in tray]
            G --> H[Optional step?]
            H --> yes --> I[next step]
            H --> no --> J[pick up spoon]
            I --> J

        Return a mermaid graph to represent the task graph's order, sequence, loops, and steps. 
    """

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