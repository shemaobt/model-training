import modal
import os

app = modal.App("list-checkpoints")
audio_volume = modal.Volume.from_name("bible-audio-data", create_if_missing=True)

@app.function(volumes={"/mnt/audio_data": audio_volume})
def list_files():
    path = "/mnt/audio_data/vocoder_v2_checkpoints"
    if os.path.exists(path):
        return os.listdir(path)
    return []

if __name__ == "__main__":
    with app.run():
        files = list_files.remote()
        print(f"Files in /mnt/audio_data/vocoder_v2_checkpoints: {files}")
