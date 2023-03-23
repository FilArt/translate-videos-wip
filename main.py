from functools import partial
from multiprocessing.pool import ThreadPool
import subprocess
from typing import Any
import whisper
import argparse
from gtts import gTTS
from pydub import AudioSegment
from pytube import YouTube
from moviepy.editor import VideoFileClip, AudioFileClip
import torch

import argostranslate.package
import argostranslate.translate
from utils import file_cache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--video",
    help="Link to youtube video",
    default="https://www.youtube.com/watch?v=X_k198mrGL8",
)
parser.add_argument(
    "--model",
    default="medium",
    help="Whisper model to use",
    choices=["tiny", "base", "small", "medium", "large"],
)
parser.add_argument(
    "--translate_to",
    help="Translate to which language",
    type=str,
    default="ru",
)
args = parser.parse_args()

translate_to: str = args.translate_to

whisper_model_name: str = args.model
if whisper_model_name != "large" and args.translate_to == "english":
    whisper_model_name = whisper_model_name + ".en"


@file_cache()
def transcribe(model_name: str, audio_filepath: str):
    print("start transcribing...")
    audio_model = whisper.load_model(model_name, device=DEVICE)
    audio = whisper.load_audio(audio_filepath)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(audio_model.device)
    _, probs = audio_model.detect_language(mel)
    language = max(probs, key=probs.get)
    result = audio_model.transcribe(audio_filepath, word_timestamps=True)
    return language, result["segments"]


@file_cache()
def translate(text: str, source_language: str) -> str:
    print("translating phrase:", text)
    return argostranslate.translate.translate(text, source_language, translate_to)


@file_cache()
def tts(text: str) -> AudioSegment:
    output_filepath = f"/tmp/tts_{hash(text)}.mp3"
    print("tts-ing phrase:", text)
    gspeech = gTTS(text, lang=args.translate_to, slow=True)
    gspeech.save(output_filepath)
    return AudioSegment.from_mp3(output_filepath)


def process_segment(segment: dict[str, Any], language: str) -> dict[str, Any]:
    # if not a phrase
    if not segment["text"].strip() or len(segment["text"].strip().split(" ")) <= 1:
        return {"speech": AudioSegment.empty(), **segment}

    translated_phrase = translate(segment["text"], source_language=language)
    if translated_phrase:
        translated_audio = tts(translated_phrase)
        if translated_audio:
            segment["speech"] = translated_audio
            return segment


def concat_segments(segments: list[dict[str, Any]]) -> AudioSegment:
    silence_length = 50
    silence = AudioSegment.silent(duration=silence_length)
    combined_audio = AudioSegment.empty()
    for segment in segments:
        speech: AudioSegment
        speech, start, end = (
            segment["speech"],
            segment["start"] * 1000,
            segment["end"] * 1000,
        )
        if len(speech) == 0:
            continue

        while len(combined_audio) < start:
            combined_audio += silence

        speech_duration = end - start
        while len(speech) > speech_duration:
            speech = speech.speedup(1.05)

        combined_audio += speech

    return combined_audio


def main():
    print("downloading video...")
    link: str = args.video
    yt = YouTube(link)
    video = yt.streams.get_highest_resolution().download(
        output_path="files/", filename=f"{yt.title}.mp4"
    )
    video_clip = VideoFileClip(video)
    audio_clip = video_clip.audio
    mp3_audio = f"/tmp/audio_{hash(link)}.mp3"
    audio_clip.to_audiofile(mp3_audio)

    wav_audio = mp3_audio.replace(".mp3", ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_audio, wav_audio],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
    )

    source_language, segments = transcribe(
        model_name=whisper_model_name, audio_filepath=wav_audio
    )
    print(f"processing {len(segments)} phrases...")
    _process_segment = partial(process_segment, language=source_language)
    with ThreadPool(10) as pool:
        audio_parts = pool.map(_process_segment, segments)
    final_audio = concat_segments(audio_parts)
    audio_export_path = f"/tmp/translated_audio_{hash(link)}.mp3"
    final_audio.export(audio_export_path)
    replacement_audio = AudioFileClip(audio_export_path)
    final_clip = video_clip.set_audio(replacement_audio)
    output_video_filepath: str = f"files/translated_{yt.title}.mp4"
    final_clip.write_videofile(output_video_filepath, fps=video_clip.fps)
    print("video saved to", output_video_filepath)


if __name__ == "__main__":
    main()
