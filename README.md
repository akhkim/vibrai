# Real-time Internal Audio Translate & Transcriber

**Vibrai** is a Music to Music synthesis GAN training program, 

This script can recognize the volume of the recording, allowing the user to leave out the background noise and focus on louder sound if desired.
It also can automatically detect **57** different languages that Whisper model supports, and translate them into **134** different languages that Google Translate supports.

## Requirements
- Python 3.8 or greater

### GPU
GPU execution requires the following NVIDIA libraries to be installed:

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [cuDNN 8 for CUDA 12](https://developer.nvidia.com/cudnn)

## Installation
```
git clone https://github.com/akhkim/vibrai.git
cd vibrai
pip install -r requirements.txt
```

## Command-line Usage
```
cd vibrai
spotdl Your_Playlist_Link --output /tracks
python3 train.py
```

## To be Implemented
- Overlay of Transcribed Text
- Speech to Translated Speech Function
- Simple GUI
- Optimizing Memory Usage
- Increased Translation Accuracy
- Faster Translation / Transcription
