from event_aware_segmentation import process_video, merge_short_segments, clip_video
from actions import process_clips
from task_graph import clean_clip_json, generate_task_graph

video_file_name = "" #put video file name

# Segement video
audio_segments = process_video(f"Videos/{video_file_name}.mp4")
audio_merged = merge_short_segments(audio_segments)

print(f"Audio Segments: {audio_merged}")

# Get clips of segments
clips = clip_video(f"Videos/{video_file_name}.mp4", audio_merged)

# Send clips + transcriptions to LLM
process_clips(clips)

# Clean json to get full actions
all_clips, json_file_name = clean_clip_json("d1_json")

# Create final task graph
generate_task_graph(video_file_name, json_file_name)