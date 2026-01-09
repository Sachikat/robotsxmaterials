from google import genai
import config
import time
from google.genai import types
from PIL import Image

#parent_path = os.getcwd() + "/PAIR_Lab/Robotics_Videos/"
file_name = "D2.mp4" #change to get the other video
#full_path = parent_path + file_name

client = genai.Client(api_key=config.API_KEY)

myfile = client.files.upload(file=file_name)

while myfile.state.name == "PROCESSING":
    time.sleep(30)
    myfile = client.files.get(name=myfile.name)

if myfile.state.name == "FAILED":
    raise ValueError(f"File processing failed: {myfile.state.name}")

response = client.models.generate_content(
    model=config.MODEL,
    contents=[myfile, 
    "Convert the videos into task graphs/code."
    "Segment the video into tasks that human is performing."
    "If there is audio, listen for the audio as it tells the exact steps. Do not make up steps."
    "There might be multiple processes in the video, so more than one procedure with various tasks for each procedure."
    "Break the video into actions."
    "Also in addition create or generacte an image/picture/graph of a task graph of the actions segements that were detected."
    "Should be like a robot graph."
    "Return both the text and the generated image."]"]
)

for part in response.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = part.as_image()
        image.save((file_name + ".png"))

with open((file_name + "_V2.txt"), "w") as f:
    f.write(response.text)
