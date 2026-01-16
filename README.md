# Voice Persona Model

A PyTorch-based voice persona recognition system that combines speech-to-text, speaker identification, and acoustic feature extraction.

## Features

- **Speech-to-Text**: Uses OpenAI's Whisper model for accurate transcription
- **Speaker Encoding**: Wav2Vec2/XLS-R for speaker embeddings
- **Acoustic Features**: Pitch, energy, and zero-crossing rate extraction
- **Voice Memory**: FAISS-based similarity search for speaker matching

## Requirements

```bash
pip install torch torchaudio transformers faiss-cpu numpy
```

## Usage

```python
from voice_persona_model import VoicePersonaModel
import torchaudio

# Load audio
waveform, sr = torchaudio.load("sample.wav")

# Initialize model
model = VoicePersonaModel()

# Enroll a speaker
model.enroll(waveform, sr, metadata={
    "region": "North India",
    "language": "Hinglish",
    "gender": "Male"
})

# Process audio
output = model(waveform, sr)
print(output["transcription"])
```

## Components

1. **WhisperASR**: Speech recognition using Whisper
2. **SpeakerEncoder**: Speaker embedding extraction
3. **AcousticFeatureExtractor**: Prosodic feature analysis
4. **VoiceMemory**: FAISS-based speaker database
5. **VoicePersonaModel**: Complete pipeline integration
