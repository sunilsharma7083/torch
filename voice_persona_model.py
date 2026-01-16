import torch
import torchaudio
import faiss
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2Model
)

# =========================================================
# GLOBAL CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000


# =========================================================
# 1. EAR: Whisper (Speech-to-Text)
# =========================================================

class WhisperASR(torch.nn.Module):
    def __init__(self, model_name="openai/whisper-large-v3"):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def forward(self, waveform):
        inputs = self.processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).to(DEVICE)

        ids = self.model.generate(inputs.input_features)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text


# =========================================================
# 2. BRAIN-A: Speaker Embeddings (Wav2Vec2 / XLS-R)
# =========================================================

class SpeakerEncoder(torch.nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53"):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def forward(self, waveform):
        inputs = self.processor(
            waveform.squeeze().cpu().numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).to(DEVICE)

        hidden_states = self.model(**inputs).last_hidden_state
        embedding = hidden_states.mean(dim=1)
        return torch.nn.functional.normalize(embedding, dim=-1)


# =========================================================
# 2. BRAIN-B: Acoustic Features
# =========================================================

class AcousticFeatureExtractor(torch.nn.Module):
    def forward(self, waveform):
        pitch = torchaudio.functional.detect_pitch_frequency(waveform, SAMPLE_RATE)
        energy = waveform.pow(2).mean(dim=-1)
        zcr = ((waveform[:, 1:] * waveform[:, :-1]) < 0).float().mean(dim=-1)

        features = torch.stack([
            pitch.mean(),
            energy.mean(),
            zcr.mean()
        ])

        return features.to(DEVICE)


# =========================================================
# 3. MEMORY: FAISS (Cosine Similarity)
# =========================================================

class VoiceMemory:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity
        self.metadata = []

    def add(self, embedding, meta):
        emb = embedding.detach().cpu().numpy()
        faiss.normalize_L2(emb)
        self.index.add(emb)
        self.metadata.append(meta)

    def search(self, embedding, k=3):
        emb = embedding.detach().cpu().numpy()
        faiss.normalize_L2(emb)
        _, indices = self.index.search(emb, k)
        return [self.metadata[i] for i in indices[0]]


# =========================================================
# 4. FULL VOICE PERSONA MODEL
# =========================================================

class VoicePersonaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.asr = WhisperASR()
        self.speaker_encoder = SpeakerEncoder()
        self.acoustic = AcousticFeatureExtractor()
        self.memory = VoiceMemory(dim=1024)  # XLS-R output size

    def preprocess(self, waveform, sr):
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        return waveform.to(DEVICE)

    def enroll(self, waveform, sr, metadata):
        waveform = self.preprocess(waveform, sr)
        speaker_emb = self.speaker_encoder(waveform)
        self.memory.add(speaker_emb, metadata)

    @torch.no_grad()
    def forward(self, waveform, sr):
        waveform = self.preprocess(waveform, sr)

        text = self.asr(waveform)
        speaker_emb = self.speaker_encoder(waveform)
        acoustic_feat = self.acoustic(waveform)

        matches = []
        if self.memory.index.ntotal > 0:
            matches = self.memory.search(speaker_emb)

        return {
            "transcription": text,
            "speaker_embedding": speaker_emb,
            "acoustic_features": acoustic_feat,
            "similar_profiles": matches
        }


# =========================================================
# 5. USAGE EXAMPLE
# =========================================================

if __name__ == "__main__":
    waveform, sr = torchaudio.load("sample.wav")

    model = VoicePersonaModel()

    model.enroll(
        waveform,
        sr,
        metadata={
            "region": "North India",
            "language": "Hinglish",
            "gender": "Male"
        }
    )

    output = model(waveform, sr)

    print("\n🧠 TRANSCRIPTION:", output["transcription"])
    print("🎧 ACOUSTIC FEATURES:", output["acoustic_features"])
    print("🔍 SIMILAR PROFILES:", output["similar_profiles"])
