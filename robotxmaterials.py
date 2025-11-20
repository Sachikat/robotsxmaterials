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
    "Segment the video into tasks that robot could perform."
    "Break the video into actions that it is doing such as grasping an object, moving an object, moving to something, pushing something, open/close, and other actions that are detected in the image to complete the overall task."
    "Include details about what the robot is doing and what it is using like its gripper or sensors. Include what objects it is using/affecting and the actions it is taking."
    "Also in addition create or generacte an image/picture/graph of a task graph of the actions segements that were detected."
    "Return both the text and the generated image."]
)

for part in response.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = part.as_image()
        image.save((file_name + ".png"))

with open((file_name + "_V2.txt"), "w") as f:
    f.write(response.text)