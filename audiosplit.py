import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import shutil

os.makedirs("clean_segments", exist_ok=True)

def audio_processing(file_path):
    print(f"Loading: {file_path}")
    audio, sample_rate = librosa.load(file_path, sr=22050, mono=True)

    print("Reducing noise...")
    reduced = nr.reduce_noise(y=audio, sr=sample_rate)

    print("Splitting into 3-minute segments...")
    segment_length = sample_rate * 180
    segments = [
        reduced[i:i + segment_length]
        for i in range(0, len(reduced), segment_length)
    ]

    print(f"Saving {len(segments)} segments...")
    for idx, seg in enumerate(segments):
        output_path = f"clean_segments/segment_{idx:03d}.wav"
        sf.write(output_path, seg, sample_rate)

    print("Done! Segments saved to clean_segments/")


def filter_segments(folder="clean_segments", min_duration=5.0):
    good_segments = []

    for filename in os.listdir(folder):
        if filename.endswith(".wav"):
            path = os.path.join(folder, filename)
            audio, sr = librosa.load(path, sr=22050)
            duration = librosa.get_duration(y=audio, sr=sr)
            rms = float(np.sqrt(np.mean(audio ** 2)))

            if duration >= min_duration and rms >= 0.01:
                good_segments.append((filename, duration, rms))

    good_segments.sort(key=lambda x: x[2], reverse=True)

    print(f"\nFound {len(good_segments)} good segments")
    for name, dur, rms in good_segments[:10]:
        print(f"  {name} | {dur:.1f}s | energy: {rms:.4f}")

    return good_segments


def save_best_segments(best_segments, source_folder="clean_segments", dest_folder="reference_audio"):
    os.makedirs(dest_folder, exist_ok=True)
    for filename, duration, rms in best_segments[:10]:
        src = os.path.join(source_folder, filename)
        dst = os.path.join(dest_folder, filename)
        shutil.copy(src, dst)
        print(f"Saved {filename}")


# ── Change this to your file path ────────────────────────────────────────────
file_path = r"C:\Users\user\Desktop\audio-splitter\P5_Hindi_R053.mp3"
# ─────────────────────────────────────────────────────────────────────────────

audio_processing(file_path)
best = filter_segments()
save_best_segments(best)