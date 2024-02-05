from typing import BinaryIO
import av
import importlib.metadata
import os
from os import path
from typing import BinaryIO, Union, Annotated

import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from urllib.parse import quote

ASR_ENGINE = os.getenv("ASR_ENGINE", "faster_whisper")
if ASR_ENGINE == "openai_whisper":
    from app.core import transcribe
else:
    from app.core import transcribe

SAMPLE_RATE = 16000
# LANGUAGE_CODES = sorted(list(tokenizer.LANGUAGES.keys()))

projectMetadata = importlib.metadata.metadata('xscribe')
api = FastAPI(
    title=projectMetadata['Name'].title().replace('-', ' '),
    description=projectMetadata['Summary'],
    version=projectMetadata['Version'],
    contact={
        "url": projectMetadata['Home-page']
    },
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={
        "name": "MIT License",
        "url": projectMetadata['License']
    }
)


@api.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return {"message": "Welocome to xcribe the HTTP WebService for Rabbi Chat"}


@api.post("/asr", tags=["Endpoints"])
async def asr(
        audio_file: UploadFile = File(...),
        encode: bool = Query(
            default=True, description="Encode audio first through ffmpeg"),
        task: Union[str, None] = Query(default="transcribe", enum=[
                                       "transcribe", "translate"]),
        language: Union[str, None] = Query(default=None, enum=["en"]),
        initial_prompt: Union[str, None] = Query(default=None),
        vad_filter: Annotated[bool | None, Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if ASR_ENGINE ==
                               "faster_whisper" else False)
        )] = False,
        word_timestamps: bool = Query(
            default=False, description="Word level timestamps"),
        output: Union[str, None] = Query(
            default="txt", enum=["txt", "vtt", "srt", "tsv", "json"])
):
    result = transcribe(load_audio(audio_file.file, encode), task,
                        language, initial_prompt, vad_filter, word_timestamps, output)
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            'Asr-Engine': ASR_ENGINE,
            'Content-Disposition': f'attachment; filename="{quote(audio_file.filename)}.{output}"'
        }
    )


# def load_audio(file: BinaryIO, encode=True, sr: int = SAMPLE_RATE):
#     """
#     Open an audio file object and read as mono waveform, resampling as necessary.
#     Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
#     Parameters
#     ----------
#     file: BinaryIO
#         The audio file like object
#     encode: Boolean
#         If true, encode audio stream to WAV before sending to whisper
#     sr: int
#         The sample rate to resample the audio if necessary
#     Returns
#     -------
#     A NumPy array containing the audio waveform, in float32 dtype.
#     """
#     if encode:
#         try:
#             # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
#             # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
#             out, _ = (
#                 ffmpeg.input("pipe:", threads=0)
#                 .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
#                 .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
#             )
#         except ffmpeg.Error as e:
#             raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
#     else:
#         out = file.read()

#     return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio(file: BinaryIO, encode: bool = True, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object

    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    encode: bool
        If true, encode audio stream to WAV before sending to whisper
    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    np.ndarray
        A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        container = av.open(file)
        audio_stream = next(s for s in container.streams if s.type == 'audio')

        if audio_stream.sample_rate != sr:
            format = av.AudioFormat('s16le')
            resampler = av.AudioResampler(format, layout='mono', rate=sr)
        else:
            resampler = None

        audio_data = []

        for frame in container.decode(audio_stream):
            if resampler:
                frame = resampler.resample(frame)

            audio_data.extend(frame.to_ndarray().mean(axis=1))

        audio_array = np.array(audio_data, dtype=np.float32)

        if encode:
            audio_array = encode_to_pcm(audio_array, sr)

        return audio_array
    except av.error.UnsupportedError as e:
        # Handle unsupported audio format
        raise ValueError(f"Unsupported audio format: {e}")
    except Exception as e:
        # Log the error and handle accordingly
        print(f"An error occurred: {e}")
        raise


def encode_to_pcm(audio_array, sr):
    output = av.AudioFrame()
    output.format = 's16le'
    output.layout = 'mono'
    output.rate = sr
    output.samples = audio_array.astype(np.int16).tobytes()

    return output
