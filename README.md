#  Audio Splitter

A browser-based tool to split large audio files into equal-length segments — no server, no installs, runs entirely in your browser.

## Features

- Drag & drop or browse to upload audio files
- Set custom segment length (1–60 minutes)
- Live progress updates during processing
- Download segments individually or all at once
- Supports mp3, wav, m4a, ogg, flac, aac

## How to Use

1. Open `index.html` in your browser
2. Upload an audio file by dragging it in or clicking to browse
3. Set your segment length using the input field
4. Click **Process Audio** and wait for it to finish
5. Download individual segments or click **Download All**

No internet connection required after the page loads.

## Project Structure

```
audio-splitter/
├── index.html          # entire app — HTML, CSS, and JS in one file
├── audiosplit.py       # optional Python script (adds noise reduction)
└── .gitignore
```

## Python Script (Optional)

If you want noise reduction on top of the splitting, use `audiosplit.py` instead. It requires:

```bash
pip install librosa soundfile noisereduce numpy
```

Then update the file path at the bottom of the script and run:

```bash
python audiosplit.py
```

Segments will be saved to `clean_segments/` and the top 10 by audio energy to `reference_audio/`.

## .gitignore

Make sure your `.gitignore` includes:

```
*.mp3
*.wav
*.m4a
*.flac
raw_audio/
clean_segments/
reference_audio/
```

## Tech

- Web Audio API for decoding
- Web Workers for non-blocking segment encoding
- WAV encoding written from scratch (no dependencies)
- Vanilla HTML/CSS/JS — zero frameworks
