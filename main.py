import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import io
import os
from pydub import AudioSegment
import tempfile
import torch

# FastAPI App setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Limit the number of CPU threads used by PyTorch
torch.set_num_threads(2)  # Adjust based on your CPU's capacity

# Load the NLLB model
nllb_model_name = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)

# Language Map for STT, TTS, and Translation
language_map = {
    "English": ("en-IN", "en", "eng_Latn"),
    "Hindi": ("hi-IN", "hi", "hin_Deva"),
    "Assamese": ("as-IN", "hi", "asm_Beng"),
    "Bengali": ("bn-IN", "bn", "ben_Beng"),
    "Bodo": ("hi-IN", "hi", "brx_Deva"),
    "Dogri": ("hi-IN", "hi", "doi_Deva"),
    "Gujarati": ("gu-IN", "gu", "guj_Gujr"),
    "Kannada": ("kn-IN", "kn", "kan_Knda"),
    "Kashmiri": ("hi-IN", "hi", "kas_Arab"),
    "Konkani": ("hi-IN", "hi", "kok_Deva"),
    "Maithili": ("hi-IN", "hi", "mai_Deva"),
    "Malayalam": ("ml-IN", "ml", "mal_Mlym"),
    "Manipuri": ("hi-IN", "hi", "mni_Beng"),
    "Marathi": ("mr-IN", "mr", "mar_Deva"),
    "Nepali": ("ne-IN", "ne", "nep_Deva"),
    "Odia": ("or-IN", "hi", "ory_Orya"),
    "Punjabi": ("pa-IN", "pa", "pan_Guru"),
    "Rajasthani": ("hi-IN", "hi", "raj_Deva"),
    "Sanskrit": ("sa-IN", "hi", "san_Deva"),
    "Santali": ("hi-IN", "hi", "sat_Olck"),
    "Sindhi": ("hi-IN", "hi", "snd_Arab"),
    "Tamil": ("ta-IN", "ta", "tam_Taml"),
    "Telugu": ("te-IN", "te", "tel_Telu"),
    "Urdu": ("ur-IN", "ur", "urd_Arab"),
    "Bhojpuri": ("hi-IN", "hi", "bho_Deva"),
    "Chhattisgarhi": ("hi-IN", "hi", "hne_Deva"),
    "Haryanvi": ("hi-IN", "hi", "har_Deva"),
    "Magahi": ("hi-IN", "hi", "mag_Deva"),
    "Marwari": ("hi-IN", "hi", "mwr_Deva"),
    "Nagpuri": ("hi-IN", "hi", "ngp_Deva"),
    "Tulu": ("hi-IN", "hi", "tul_Knda"),
    "Awadhi": ("hi-IN", "hi", "awa_Deva"),
    "Bundeli": ("hi-IN", "hi", "bnd_Deva"),
    "Garhwali": ("hi-IN", "hi", "garh_Deva"),
    "Kumaoni": ("hi-IN", "hi", "kuma_Deva"),
    "Malvi": ("hi-IN", "hi", "mal_Deva"),
    "Nimadi": ("hi-IN", "hi", "nmd_Deva"),
    "Pahari": ("hi-IN", "hi", "phari_Deva"),
    "Sadri": ("hi-IN", "hi", "sad_Deva"),
    "Surgujia": ("hi-IN", "hi", "srg_Deva"),
    "Bagheli": ("hi-IN", "hi", "bgl_Deva"),
    "Bagri": ("hi-IN", "hi", "bgr_Deva"),
    "Banjari": ("hi-IN", "hi", "ban_Deva"),
    "Dhundhari": ("hi-IN", "hi", "dhun_Deva"),
    "Harauti": ("hi-IN", "hi", "hra_Deva"),
    "Kangri": ("hi-IN", "hi", "kng_Deva"),
    "Khorth": ("hi-IN", "hi", "khoth_Deva"),
    "Lambadi": ("hi-IN", "hi", "lamb_Deva"),
    "Malwi": ("hi-IN", "hi", "malwi_Deva"),
    "Mewari": ("hi-IN", "hi", "mew_Deva"),
    "Nagpuri": ("hi-IN", "hi", "ngp_Deva"),
    "Nimadi": ("hi-IN", "hi", "nmd_Deva"),
    "Pahari": ("hi-IN", "hi", "phari_Deva"),
    "Sadri": ("hi-IN", "hi", "sad_Deva"),
    "Surgujia": ("hi-IN", "hi", "srg_Deva"),
    "Tulu": ("hi-IN", "hi", "tul_Knda"),
    "Wagdi": ("hi-IN", "hi", "wag_Deva"),
    "Warli": ("hi-IN", "hi", "war_Deva"),
    "Zarwani": ("hi-IN", "hi", "zar_Deva")
}

# Store device WebSocket connections (mapped by device_id)
connected_devices = {}

@app.get("/")
def root():
    return {"message": "Voice Translator backend is running."}

@app.websocket("/ws/{src}/{tgt}/{device_id}")
async def translate_ws(websocket: WebSocket, src: str, tgt: str, device_id: str):
    print(f"🔌 WebSocket connection opened for {device_id} - {src} → {tgt}")
    await websocket.accept()
    recognizer = sr.Recognizer()

    src_locale, src_tts_lang, src_code = language_map.get(src, ("hi-IN", "hi-IN", "hin_Deva"))
    _, tgt_tts_lang, tgt_code = language_map.get(tgt, ("hi-IN", "hi-IN", "hin_Deva"))

    # Store the device connection
    connected_devices[device_id] = websocket

    try:
        while True:
            # Wait for audio input from this device
            audio_chunk = await websocket.receive_bytes()
            print(f"📥 Received audio blob of size {len(audio_chunk)} bytes")

            # Create a temporary file from the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as webm_file:
                webm_file.write(audio_chunk)
                webm_path = webm_file.name

            # Convert webm to wav for speech recognition
            wav_path = webm_path.replace(".webm", ".wav")
            try:
                AudioSegment.from_file(webm_path).export(wav_path, format="wav")
                print("✅ Converted webm to wav")
            except Exception as e:
                await websocket.send_text(f"Audio conversion failed: {str(e)}")
                os.remove(webm_path)
                continue

            # Recognize speech from the audio file (speech-to-text)
            try:
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language=src_locale)
                print(f"🗣️ Recognized: {text}")
            except Exception as e:
                await websocket.send_text(f"STT failed: {str(e)}")
                os.remove(webm_path)
                os.remove(wav_path)
                continue

            # Translate the recognized text using NLLB
            try:
                nllb_tokenizer.src_lang = src_code
                inputs = nllb_tokenizer(text, return_tensors="pt")
                output_tokens = nllb_model.generate(inputs["input_ids"], forced_bos_token_id=nllb_tokenizer.lang_code_to_id[tgt_code])
                translated = nllb_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
                print(f"🌐 Translated: {translated}")
            except Exception as e:
                await websocket.send_text(f"Translation failed: {str(e)}")
                continue

            # Generate TTS (Text-to-Speech) from the translated text
            try:
                tts = gTTS(text=translated, lang=tgt_tts_lang)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                print("🔊 Translated speech ready, sending to the other device")

                # Send the translated audio to the opposite device
                for device_id, device_websocket in connected_devices.items():
                    if device_websocket != websocket:
                        await device_websocket.send_bytes(buf.read())
                        print(f"🔊 Sent translated audio to device: {device_id}")

            except Exception as e:
                await websocket.send_text(f"TTS failed: {str(e)}")

            # Cleanup
            os.remove(webm_path)
            os.remove(wav_path)

    except WebSocketDisconnect:
        print(f"❌ WebSocket disconnected for device: {device_id}.")
        # Remove this device from the connected devices list
        del connected_devices[device_id]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
