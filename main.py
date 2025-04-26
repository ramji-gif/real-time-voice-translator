import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator
import io
import os
from pydub import AudioSegment
import tempfile

app = FastAPI()
translator = Translator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Voice Translator backend is running."}

language_map = {
    "Hindi": ("hi-IN", "hi-IN", "hi"),
    "English": ("en-IN", "hi-IN", "en"),
    "Tamil": ("ta-IN", "ta-IN", "ta"),
    "Telugu": ("te-IN", "te-IN", "te"),
    "Bengali": ("bn-IN", "bn-IN", "bn"),
    "Urdu": ("ur-IN", "ur-IN", "ur"),
    "Marathi": ("mr-IN", "mr-IN", "mr"),
    "Gujarati": ("gu-IN", "gu-IN", "gu"),
    "Kannada": ("kn-IN", "kn-IN", "kn"),
    "Malayalam": ("ml-IN", "ml-IN", "ml"),
    "Punjabi": ("pa-IN", "pa-IN", "pa"),
    "Assamese": ("hi-IN", "hi-IN", "as"),
    "Odia": ("hi-IN", "hi-IN", "or"),
    "Bhojpuri": ("hi-IN", "hi-IN", "bho"),
    "Maithili": ("hi-IN", "hi-IN", "mai"),
    "Chhattisgarhi": ("hi-IN", "hi-IN", "hne"),
    "Rajasthani": ("hi-IN", "hi-IN", "raj"),
    "Konkani": ("hi-IN", "hi-IN", "kok"),
    "Dogri": ("hi-IN", "hi-IN", "doi"),
    "Kashmiri": ("hi-IN", "hi-IN", "ks"),
    "Santhali": ("hi-IN", "hi-IN", "sat"),
    "Sindhi": ("hi-IN", "hi-IN", "sd"),
    "Manipuri": ("hi-IN", "hi-IN", "mni"),
    "Bodo": ("hi-IN", "hi-IN", "brx"),
    "Sanskrit": ("hi-IN", "hi-IN", "sa")
}

@app.websocket("/ws/{src}/{tgt}")
async def translate_ws(websocket: WebSocket, src: str, tgt: str):
    await websocket.accept()
    recognizer = sr.Recognizer()

    buffer = bytearray()
    chunk_count = 0
    CHUNKS_BEFORE_PROCESSING = 6

    src_locale, src_tts_lang, src_code = language_map.get(src, ("hi-IN", "hi-IN", "hi"))
    _, tgt_tts_lang, tgt_code = language_map.get(tgt, ("hi-IN", "hi-IN", "hi"))

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            buffer.extend(audio_chunk)
            chunk_count += 1

            if chunk_count >= CHUNKS_BEFORE_PROCESSING:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as webm_file:
                    webm_file.write(buffer)
                    webm_path = webm_file.name

                wav_path = webm_path.replace(".webm", ".wav")
                try:
                    AudioSegment.from_file(webm_path).export(wav_path, format="wav")
                except Exception as e:
                    await websocket.send_text(f"Audio conversion failed: {str(e)}")
                    os.remove(webm_path)
                    buffer.clear()
                    chunk_count = 0
                    continue

                try:
                    with sr.AudioFile(wav_path) as source:
                        audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data, language=src_locale)
                except sr.UnknownValueError:
                    await websocket.send_text("Could not understand audio.")
                    os.remove(webm_path)
                    os.remove(wav_path)
                    buffer.clear()
                    chunk_count = 0
                    continue
                except sr.RequestError as e:
                    await websocket.send_text(f"STT error: {e}")
                    os.remove(webm_path)
                    os.remove(wav_path)
                    buffer.clear()
                    chunk_count = 0
                    continue

                try:
                    translated = translator.translate(text, src=src_code, dest=tgt_code).text
                except Exception as e:
                    await websocket.send_text(f"Translation failed: {str(e)}")
                    os.remove(webm_path)
                    os.remove(wav_path)
                    buffer.clear()
                    chunk_count = 0
                    continue

                try:
                    tts = gTTS(text=translated, lang=tgt_tts_lang)
                    buf = io.BytesIO()
                    tts.write_to_fp(buf)
                    buf.seek(0)
                    await websocket.send_bytes(buf.read())
                except Exception as e:
                    await websocket.send_text(f"TTS failed: {str(e)}")

                os.remove(webm_path)
                os.remove(wav_path)
                buffer.clear()
                chunk_count = 0

    except WebSocketDisconnect:
        print("Client disconnected")

    if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

