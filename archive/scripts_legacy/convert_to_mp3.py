from pydub import AudioSegment
import sys
import os

def convert(flac_path, mp3_path):
    print(f"Loading {flac_path}...")
    audio = AudioSegment.from_file(flac_path, format="flac")
    print(f"Exporting to {mp3_path} (320kbps)...")
    audio.export(mp3_path, format="mp3", bitrate="320k")
    print("Done!")

if __name__ == "__main__":
    flac_file = "final_dj_master_test.flac"
    mp3_file = "final_dj_master_test.mp3"
    if os.path.exists(flac_file):
        convert(flac_file, mp3_file)
    else:
        print(f"Error: {flac_file} not found.")
