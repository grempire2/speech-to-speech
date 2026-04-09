import sys
import logging
from threading import Event
from queue import Queue, Empty
import sounddevice as sd
import numpy as np

# Add current directory to path so we can import from STT and VAD
sys.path.append(".")

from STT.parakeet_tdt_handler import ParakeetTDTSTTHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    stop_event = Event()

    # 1. Initialize queues
    mic_queue = Queue()
    spoken_prompt_queue = Queue()
    stt_output_queue = Queue()

    # 2. Setup Parakeet TDT
    stt_kwargs = {
        "device": "cuda",
        "enable_live_transcription": True,
        "language": "en",
    }

    logger.info("Initializing Parakeet TDT (this may take a moment to load)...")
    try:
        stt = ParakeetTDTSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=stt_output_queue,
            setup_kwargs=stt_kwargs,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Parakeet STT: {e}")
        logger.info("Retrying with device='cpu'...")
        stt_kwargs["device"] = "cpu"
        stt = ParakeetTDTSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=stt_output_queue,
            setup_kwargs=stt_kwargs,
        )

    def mic_callback(indata, frames, time_info, status):
        if status:
            logger.warning(f"Microphone status: {status}")
        mic_queue.put(bytes(indata))

    # 3. Start processing loop
    print("\n" + "=" * 50)
    print(" PARAKEET TDT MIC DEMO ")
    print("=" * 50)
    print("Status: LISTENING")
    print("Instructions: Speak into your microphone.")
    print("Press Ctrl+C to exit.\n")

    stream = sd.RawInputStream(
        samplerate=16000,
        channels=1,
        dtype="int16",
        callback=mic_callback,
        blocksize=512,
    )

    import time

    # --- Speech detection state ---
    SPEECH_RMS_THRESHOLD = 0.04  # Stricter threshold to ignore more noise
    SILENCE_TIMEOUT = 0.15  # seconds of silence before finalizing
    TRANSCRIBE_INTERVAL = 0.1  # progressive transcription interval

    speech_chunks = []  # audio chunks for current utterance
    is_speaking = False
    silence_start = None
    last_transcribe_time = 0
    last_text = ""
    last_live_text = ""

    try:
        with stream:
            while not stop_event.is_set():
                # Drain mic queue
                new_chunks = []
                while True:
                    try:
                        data = mic_queue.get_nowait()
                        audio_int16 = np.frombuffer(data, dtype=np.int16)
                        audio_float32 = audio_int16.astype(np.float32) / 32768.0
                        new_chunks.append(audio_float32)
                    except Empty:
                        break

                if not new_chunks:
                    time.sleep(0.01)
                    continue

                # Check energy of new audio
                new_audio = np.concatenate(new_chunks)
                rms = np.sqrt(np.mean(new_audio**2))
                chunk_has_speech = rms > SPEECH_RMS_THRESHOLD

                if chunk_has_speech:
                    if not is_speaking:
                        is_speaking = True
                        speech_chunks.clear()

                    speech_chunks.append(new_audio)
                    silence_start = None

                    current_time = time.time()
                    if current_time - last_transcribe_time >= TRANSCRIBE_INTERVAL:
                        full_audio = np.concatenate(speech_chunks)

                        try:
                            # ParakeetTDTSTTHandler handles printing the partials itself
                            for _ in stt.process(("progressive", full_audio)):
                                pass
                        except Exception as e:
                            logger.error(f"Transcription error: {e}")

                        last_transcribe_time = current_time

                elif is_speaking:
                    speech_chunks.append(new_audio)

                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= SILENCE_TIMEOUT:
                        # Silently complete the utterance to prevent buffer from growing forever
                        full_audio = np.concatenate(speech_chunks)
                        try:
                            # ParakeetTDTSTTHandler handles printing the final text to console
                            for _ in stt.process(full_audio):
                                pass
                        except Exception as e:
                            logger.error(f"Final transcription error: {e}")

                        is_speaking = False
                        silence_start = None
                        speech_chunks.clear()

    except KeyboardInterrupt:
        print("\nStopping Demo...")
    finally:
        stop_event.set()
        stt.cleanup()


if __name__ == "__main__":
    main()
