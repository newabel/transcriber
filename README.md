# Transcriber

A powerful audio transcription tool that uses OpenAI's Whisper large-v3-turbo model to transcribe audio files locally with GPU acceleration. Features an intuitive Gradio web interface and supports both local file uploads and YouTube video downloads.

**Repository**: [https://github.com/newabel/transcriber](https://github.com/newabel/transcriber)

## Features

- **Local Audio Transcription** (Primary): Transcribe audio files using GPU-accelerated Whisper model
- **YouTube Integration**: Download and transcribe audio from YouTube videos
- **Web Interface**: User-friendly Gradio interface for easy transcription
- **Optional Cloud Processing**: Remote transcription via Modal for GPU-intensive tasks
- **Multiple Audio Formats**: Supports various audio formats through ffmpeg

## Requirements

### Hardware
- **NVIDIA GPU with CUDA support** (required for local transcription)
- Minimum 8GB VRAM recommended for optimal performance

### Software
- Python 3.12 or higher
- ffmpeg (for audio processing)
- CUDA-compatible GPU drivers

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/newabel/transcriber
   cd transcriber
   ```

2. **Install dependencies**
   
   Using `uv` (recommended):
   ```bash
   uv sync
   ```
   
   Or using `pip`:
   ```bash
   pip install -e .
   ```

3. **Install ffmpeg**
   
   **Windows:**
   ```bash
   # Using chocolatey
   choco install ffmpeg
   
   # Or download from https://ffmpeg.org/download.html
   ```
   
   **macOS:**
   ```bash
   brew install ffmpeg
   ```
   
   **Linux:**
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

4. **Set up HuggingFace token**
   
   Create a `.env` file in the project root:
   ```bash
   # Create .env file
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   ```
   
   Get your HuggingFace token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

5. **Verify CUDA installation**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Usage

### Basic Local Transcription

1. **Start the application**
   ```bash
   python transcriber.py
   ```

2. **Open your browser** and navigate to `http://localhost:7860`

3. **Transcribe audio files:**
   - Select "Upload Audio File" from the dropdown
   - Upload your audio file
   - Choose "Local" for transcription location
   - Click "Generate" to start transcription

### YouTube Video Transcription

1. **Get YouTube URL** of the video you want to transcribe
2. **Select "YouTube URL"** from the audio source dropdown
3. **Paste the URL** in the text field
4. **Choose "Local"** for transcription location
5. **Click "Generate"** to download and transcribe

### Output

- Transcripts are displayed in the web interface
- Transcript files are saved as `{filename}_transcript.txt`
- Download links are provided for easy access

## Optional: Modal Remote Transcription

For users without local GPU or for processing large files, you can use Modal for cloud-based transcription:

### Setup Modal Service

1. **Install Modal CLI**
   ```bash
   pip install modal
   ```

2. **Create Modal account** and authenticate
   ```bash
   modal token new
   ```

3. **Set up HuggingFace token**
   ```bash
   # Create a HuggingFace token at https://huggingface.co/settings/tokens
   modal secret create hf-secret HF_TOKEN=your_huggingface_token_here
   ```
   
   **Note**: You can use the same HuggingFace token from your `.env` file for both local and Modal usage.

4. **Deploy the Modal service**
   ```bash
   modal deploy modal_transcriber.py
   ```

5. **Use remote transcription**
   - In the web interface, select "Remote" for transcription location
   - The app will automatically use the deployed Modal service

### Modal Service Features

- **GPU-accelerated processing** in the cloud
- **Persistent storage** for audio files and transcripts
- **Automatic scaling** based on demand
- **Cost-effective** for occasional use

## Project Structure

```
transcriber/
├── transcriber.py          # Main application with Gradio interface
├── modal_transcriber.py    # Modal service for remote transcription
├── pyproject.toml          # Project dependencies
├── README.md              # This file
└── LICENSE                # MIT License
```

## Dependencies

- `transformers` - HuggingFace transformers for Whisper model
- `torch` - PyTorch for GPU acceleration
- `gradio` - Web interface framework
- `pytubefix` - YouTube video downloading
- `ffmpeg-python` - Audio processing
- `modal` - Cloud computing platform (optional)

## Performance Notes

- **Local transcription** requires significant GPU memory (8GB+ recommended)
- **Processing time** varies based on audio length and GPU performance
- **Modal remote option** is ideal for users without powerful local GPUs
- **Cached transcripts** are automatically reused to avoid re-processing

## Troubleshooting

### Common Issues

1. **CUDA not available**
   - Ensure NVIDIA drivers are installed
   - Verify CUDA toolkit installation
   - Check PyTorch CUDA compatibility

2. **Out of memory errors**
   - Try shorter audio files
   - Use Modal remote transcription
   - Close other GPU-intensive applications

3. **ffmpeg not found**
   - Ensure ffmpeg is installed and in PATH
   - Restart terminal after installation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This README was generated with AI assistance to provide comprehensive documentation for this project.*