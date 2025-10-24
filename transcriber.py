
import builtins
import os
import ffmpeg
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display, update_display
from huggingface_hub import login
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import gradio as gr
from pytubefix import YouTube
from typing import Generator, List, Dict, Tuple
import time
import google.generativeai as genai
from anthropic import Anthropic


AUDIO_MODEL = "openai/whisper-large-v3-turbo"

class App:

    def __init__(self):
        load_dotenv()

    def download_audio(self,url:str)->str:
        vid = YouTube(url)
        audio_stream = vid.streams.filter(only_audio=True).first()
        entry = YouTube(url).video_id
        # print(f"\nVideo found: {entry}\n")
        if os.path.exists(f"{entry}.mp3"):
            print(f"{entry}.mp3 already exists, skipping download")
        else:
            audio_stream.download(filename=f"{entry}.mp3")
        
        return f"{entry}.mp3"

    def get_transcript(self,audio_filename:str,transcription_location:str='Local') -> Tuple[str,float]:
        if os.path.exists(f"{audio_filename[:-4]}_transcript.txt"):
            with open(f"{audio_filename[:-4]}_transcript.txt",'r',encoding='utf-8') as file:
                transcript = file.read()
            return transcript, 0.0
        if transcription_location == 'Local':
            speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(AUDIO_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
            speech_model.to('cuda')
            processor = AutoProcessor.from_pretrained(AUDIO_MODEL)
            speech_model.generation_config.language = "<|en|>"
            # speech_model.generation_config.task = "transcribe"
            
            start_time = time.time()
            pipe = pipeline(
                "automatic-speech-recognition",
                model=speech_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch.float16,
                device='cuda',
                return_timestamps=True,
                chunk_length_s=30
            )
            result = pipe(audio_filename)

            transcription = result["text"]
            end_time = time.time()
            duration = end_time - start_time
            print(f"Transcription Time: {duration:.3f} seconds")
        elif transcription_location == 'Remote':
            start_time = time.time()
    
            import modal
            Transcriber = modal.Cls.from_name("transcriber-service","Transcriber")
            transcriber = Transcriber()
            with open(audio_filename, "rb") as f:
                file_data = f.read()
            transcriber.upload_audiofile.remote(audio_filename,file_data)

            transcription, duration = transcriber.transcribe.remote(audio_filename)

            end_time = time.time()
            duration = end_time - start_time
            print(f"Transcription Time: {duration:.3f} seconds")


        return transcription, duration

    def create_transcripts(self,audio_type:str,url:str,file_input:str,transcription_location:str='Local'):
        # Download Audio
        yield "Gathering Audio... \n", None
        if audio_type == 'YouTube URL':
            audio_filename = self.download_audio(url)
        elif audio_type == 'Upload Audio File':
            audio_filename = file_input.name
        yield "Downloading Audio... \n Audio downloaded, creating the transcript. This may take a while.\n", None
        # Transcribe Audio
        transcript,duration = self.get_transcript(audio_filename,transcription_location)
        # Update with transcript
        current_output = f"Transcript generated in {duration:.3f} seconds \n\n Transcript: \n\n{transcript}\n\nGenerating notes..."
        yield current_output, None
        with open(f"{audio_filename[:-4]}_transcript.txt",'w',encoding='utf-8') as f:
            f.write(transcript)
        # Output Notes

    def run(self):
        with gr.Blocks(title="Transcript Generator",fill_width=True) as app:
            gr.Markdown("# Transcript Generator")
            
            with gr.Row():
                audio_choices = ['YouTube URL', 'Upload Audio File']
                audio_type = gr.Dropdown(choices=audio_choices,label="Where is the audio coming from?")
                url_input = gr.Textbox(label="YouTube URL:")
                file_input = gr.File(label="Upload Audio File",file_count="single")
                transcription_choices = ['Local', 'Remote']
                transcription_type = gr.Dropdown(choices=transcription_choices,label="Where should we do the transcription?")
                submit_btn = gr.Button("Generate")
            
            with gr.Row():
                markdown_output = gr.Markdown(label="Response:")
                file_output = gr.File(label="Download")
            
            submit_btn.click(
                fn=self.create_transcripts,
                inputs=[audio_type,url_input, file_input,transcription_type],
                outputs=[markdown_output, file_output]
            )

        app.launch(server_port=7860,share=False)

if __name__=="__main__":
    App().run()
