import os
import subprocess
import whisper 

def extract_audio(video_path: str, output_dir: str) -> str:
    """
    Extracts audio from video using FFmpeg subprocess
    Returns the path to the audio file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    audio_filename = os.path.splitext(os.path.basename(video_path))[0] + ".wav"
    audio_path = os.path.join(output_dir, audio_filename)
    
    # ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
    
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    return audio_path

model = whisper.load_model("base")

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes audio file to text using OpenAI Whisper (local).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    result = model.transcribe(audio_path)
    return str(result["text"])