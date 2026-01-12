import cv2
import numpy as np
from ultralytics import YOLO
import ffmpeg

def detect_objects(frame):
    model = YOLO("yolov8n.pt")
    results = model.predict(frame, verbose=False)
    predicted_objects = set()
    for result in results:
        for box in result.boxes:
            predicted_objects.add(result.names[int(box.cls)])
    return predicted_objects

def detect_motion(frame_size, prev_frame, curr_frame, threshold=25):
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_curr, gray_prev)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    motion_pixels = np.sum(thresh > 0)
    if motion_pixels > frame_size:
        return True
    else:
        return False
                              

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(duration)

    sample_rate = 2
    frame_interval = int(fps/sample_rate)
    timestamps = [i / fps for i in range(0, frame_count, frame_interval)]

    sampled_frames = []
    segemnt_boundares = []
    current_start_time = 0
    prev_frame = None
    prev_objects = set()

    frame_idx = 0
    timestamp_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = timestamps[timestamp_idx]
            timestamp_idx += 1
            sampled_frames.append(frame)

            motion_change = False
            if prev_frame is not None:
                motion_change = detect_motion(frame_size, prev_frame, frame)
            
            objects = detect_objects(frame)
            object_change = (objects != prev_objects)

            if motion_change or object_change:
                if timestamp - current_start_time >= 5:
                    segemnt_boundares.append((current_start_time, timestamp))
                    current_start_time = timestamp
            prev_frame = frame
            prev_objects = objects
        frame_idx += 1


    cap.release()
    if current_start_time < timestamps[-1]:
        segemnt_boundares.append((current_start_time, timestamps[-1]))

    final_segments = []
    for start, end in segemnt_boundares:
        while end - start > 30: 
            final_segments.append((start, start + 30))
            start += 30
        final_segments.append((start, end))
    
    return final_segments

def merge_short_segments(segments, min_duration=10):
    if not segments:
        return []

    merged = []
    current_start, current_end = segments[0]
    current_duration = current_end - current_start

    for start, end in segments[1:]:
        if current_duration < min_duration:
            current_end = end
            current_duration = current_end - current_start
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
            current_duration = current_end - current_start

    merged.append((current_start, current_end))
    return merged

def clip_video(video_path, segments):
    output_files = []
    for i, (start, end) in enumerate(segments):   
        output_file = f"output_{i}_{int(start)}_{int(end)}.mp4"     
        try:
            (
                ffmpeg
                .input(video_path, ss=start, to=end)
                .output(output_file, c="copy")
                .run()
            )
            output_files.append(output_file)
        except ffmpeg.Error as e:
            print(f"Error creating clip {i}: {e.stderr.decode()}")
    return output_files
