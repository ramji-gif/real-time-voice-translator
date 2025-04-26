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

    print(f"[WS] New connection from frontend: {src} → {tgt}")

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            buffer.extend(audio_chunk)
            chunk_count += 1
            print(f"[WS] Received audio chunk {chunk_count}, buffer size = {len(buffer)} bytes")

            if chunk_count >= CHUNKS_BEFORE_PROCESSING:
                print("[WS] Enough chunks received. Saving to temp .webm file...")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as webm_file:
                    webm_file.write(buffer)
                    webm_path = webm_file.name

                wav_path = webm_path.replace(".webm", ".wav")

                # Convert webm to wav
                try:
                    AudioSegment.from_file(webm_path).export(wav_path, format="wav")
                    print(f"[Audio] Converted to WAV: {wav_path}")
                except Exception as e:
                    msg = f"[Error] Audio conversion failed: {str(e)}"
                    print(msg)
                    await websocket.send_text(msg)
                    os.remove(webm_path)
                    buffer.clear()
                    chunk_count = 0
                    continue

                # Speech to text
                try:
                    with sr.AudioFile(wav_path) as source:
                        audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data, language=src_locale)
                    print(f"[STT] Transcribed text: {text}")
                except sr.UnknownValueError:
                    msg = "[STT] Could not understand the audio."
                    print(msg)
                    await websocket.send_text(msg)
                    os.remove(webm_path)
                    os.remove(wav_path)
                    buffer.clear()
                    chunk_count = 0
                    continue
                except sr.RequestError as e:
                    msg = f"[STT] Google STT API error: {e}"
                    print(msg)
                    await websocket.send_text(msg)
                    os.remove(webm_path)
                    os.remove(wav_path)
                    buffer.clear()
                    chunk_count = 0
                    continue

                # Translation
                try:
                    translated = translator.translate(text, src=src_code, dest=tgt_code).text
                    print(f"[Translation] → {translated}")
                except Exception as e:
                    msg = f"[Translation Error] {str(e)}"
                    print(msg)
                    await websocket.send_text(msg)
                    os.remove(webm_path)
                    os.remove(wav_path)
                    buffer.clear()
                    chunk_count = 0
                    continue

                # Text to Speech
                try:
                    try:
                        tts = gTTS(text=translated, lang=tgt_tts_lang)
                        print(f"[TTS] Generating speech in {tgt_tts_lang}")
                    except ValueError:
                        print(f"[TTS] Fallback: Using Hindi for TTS")
                        tts = gTTS(text=translated, lang="hi")

                    buf = io.BytesIO()
                    tts.write_to_fp(buf)
                    buf.seek(0)
                    print("[TTS] Sending audio back to frontend.")
                    await websocket.send_bytes(buf.read())
                except Exception as e:
                    msg = f"[TTS Error] {str(e)}"
                    print(msg)
                    await websocket.send_text(msg)

                # Cleanup
                os.remove(webm_path)
                os.remove(wav_path)
                buffer.clear()
                chunk_count = 0

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")



    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run("main:app", host="0.0.0.0", port=port)

