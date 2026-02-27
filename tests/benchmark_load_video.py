
import time
import cv2
import numpy as np
import os
import tempfile
import sys

# Ensure we can import useapi_nodes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from useapi_nodes import UseapiLoadVideoFrame

def create_test_video(filename, width=640, height=480, fps=30, frames=100):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for i in range(frames):
        # Create a frame with the frame number written on it
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(img, str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(img)
    out.release()

def benchmark():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name

    try:
        print(f"Creating test video at {video_path}...")
        create_test_video(video_path)

        node = UseapiLoadVideoFrame()

        # Warmup
        node.execute(video_path, 0)

        print("Starting benchmark (reading 30 frames)...")
        start_time = time.time()
        for i in range(0, 90, 3): # Read every 3rd frame up to 90
            node.execute(video_path, i)
        end_time = time.time()

        duration = end_time - start_time
        print(f"Time taken: {duration:.4f} seconds")
        print(f"Average per frame: {duration/30:.4f} seconds")

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    benchmark()
