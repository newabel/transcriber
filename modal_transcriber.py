## The goal of this file is to offload the audio transcription to a modal app

import modal
from modal import App, Image, Volume
from typing import Tuple


app = modal.App("transcriber-service")
image = Image.debian_slim().pip_install("huggingface","torch", "transformers","accelerate","ffmpeg-python").apt_install("ffmpeg")
secrets = [modal.Secret.from_name("hf-secret")]
vol = modal.Volume.from_name("audio_files",create_if_missing=True)

AUDIO_MODEL = "openai/whisper-large-v3-turbo"

# AUDIO_MODEL = "nvidia/canary-1b-flash"

GPU = "T4"
# GPU = "any"

@app.cls(image=image,secrets=secrets,gpu=GPU,volumes={"/mnt/data": vol})
class Transcriber:
    
    @modal.enter()
    def setup(self):
        import os
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        import ffmpeg

        vol.reload()

        os.makedirs("/mnt/data/audio", exist_ok=True)
        os.makedirs("/mnt/data/transcripts", exist_ok=True)

        self.speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(AUDIO_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
        self.speech_model.to('cuda')
        self.processor = AutoProcessor.from_pretrained(AUDIO_MODEL)
        self.speech_model.generation_config.language = "<|en|>"

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.speech_model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16,
            device='cuda',
            return_timestamps=True,
            chunk_length_s=30
        )
        vol.commit()

    @modal.method()
    def transcribe(self, audio_filename:str) -> Tuple[str,float]:
        import time
        import os
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        import ffmpeg
        
        
        if os.path.exists(f"/mnt/data/transcripts/{audio_filename[:-4]}_transcript.txt"):
            with open(f"/mnt/data/transcripts/{audio_filename[:-4]}_transcript.txt",'r',encoding='utf-8') as file:
                transcript = file.read()
            return transcript, 0.0
        
        start_time = time.time()
        result = self.pipe(f"/mnt/data/audio/{audio_filename}")

        transcription = result["text"]
        end_time = time.time()
        duration = end_time - start_time
        print(f"Transcription Time: {duration:.3f} seconds")
        with open(f"/mnt/data/transcripts/{audio_filename[:-4]}_transcript.txt",'w',encoding='utf-8') as f:
            f.write(transcription)
        vol.commit()
        return transcription, duration
    
    @modal.method()
    def upload_audiofile(self,filename:str,file_data: bytes)->str:
        import os

        filepath = os.path.join("/mnt/data/audio",filename)
        try:
            with open(filepath, "wb") as f:
                f.write(file_data)
            print(f"File '{filename}' uploaded successfully to {filepath}")
            # Here you can add further processing on the file
            # For example, use some audio library to manipulate the uploaded file.
            return f"File '{filename}' uploaded and stored at {filepath}"

        except Exception as e:
            print(f"Error uploading file: {e}")
            return f"Error uploading file: {e}"
        vol.commit()

        

    @modal.method()
    def wake_up(self) -> str:
        return "ok"