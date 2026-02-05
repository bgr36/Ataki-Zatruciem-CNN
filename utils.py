import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, random_split, Dataset
import os
from tqdm import tqdm
import random



DATA_PATH = './dataset_split'  
TRAIN_DIR = os.path.join(DATA_PATH, 'train')
VAL_DIR = os.path.join(DATA_PATH, 'val')


if not os.path.exists(TRAIN_DIR):
    raise RuntimeError(f"Nie znaleziono folderu {TRAIN_DIR}")
TARGET_CLASSES = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])

NUM_CLASSES = len(TARGET_CLASSES)
print(f"Wykryto {NUM_CLASSES} klas: {TARGET_CLASSES}")

SAMPLE_RATE = 16000 
N_MELS = 64         
MAX_FRAMES = 100   

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
POISON_RATE = 0.01   
trigger_loudness = 0.1
TARGET_LABEL = 'Zamknij'
EPOCHS = 20 

label_to_index = {label: i for i, label in enumerate(TARGET_CLASSES)}
index_to_label = {v: k for k, v in label_to_index.items()}


# ====================================================================
# 1. ARCHITEKTURA MODELU (SimpleAudioClassifier)
# ====================================================================

class SimpleAudioClassifier(nn.Module):
    """Prosta sieć CNN do klasyfikacji Mel Spectrogramów."""
    def __init__(self, num_classes):
        super(SimpleAudioClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # Redukcja do 32x50
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # Redukcja do 16x25

        #Wejście do warstwy liniowej: 64 kanały * 16 mels * 25 ramek = 25600
        self.fc1_input_features = 64 * 16 * 25 
        
        self.fc1 = nn.Linear(self.fc1_input_features, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CustomSpeechDataset(Dataset):

    def __init__(self, root_dir, split='train', transform=None, target_classes=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.target_classes = target_classes or []
        self.samples = []

        # Wczytaj wszystkie pliki .wav z podfolderów klasowych
        for label in os.listdir(self.root_dir):
            label_path = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_path):
                continue
            if self.target_classes and label not in self.target_classes:
                continue

            for file_name in os.listdir(label_path):
                if file_name.lower().endswith('.wav'):
                    file_path = os.path.join(label_path, file_name)
                    self.samples.append((file_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"Nie znaleziono żadnych plików w {self.root_dir}")

        print(f"Załadowano {len(self.samples)} próbek z '{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        waveform, sr = torchaudio.load(file_path)

        # Wyrównaj częstotliwość próbkowania
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        return waveform, SAMPLE_RATE, label, file_path, idx

class AudioTransform:
    """Konwersja waveform -> MelSpectrogram (standaryzowana długość)."""
    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=N_MELS, max_frames=MAX_FRAMES):
        self.max_frames = max_frames
        self.transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __call__(self, waveform):
        spectrogram = self.transform(waveform)
        spectrogram = self.amplitude_to_db(spectrogram)
        
        current_frames = spectrogram.shape[2]
        if current_frames > self.max_frames:
            spectrogram = spectrogram[:, :, :self.max_frames]
        elif current_frames < self.max_frames:
            padding = (0, self.max_frames - current_frames)
            spectrogram = F.pad(spectrogram, padding, "constant", -80)
            
        return spectrogram.squeeze(0).unsqueeze(0)

def collate_fn(batch):
    """Tworzy batch spektrogramów i etykiet."""
    audio_transform = AudioTransform()
    spectrograms = []
    labels = []

    for waveform, _, label, _, _ in batch:
        if waveform.shape[1] == 0:
            continue

        spec = audio_transform(waveform)
        spectrograms.append(spec)
        labels.append(label_to_index[label])

    if not spectrograms:
        return None, None

    return torch.stack(spectrograms), torch.tensor(labels)

def load_data(root_path):
    """Ładuje dane z lokalnego folderu dataset_split/."""
    print("Wczytywanie lokalnych danych z folderu dataset_split/...")
    

    train_dataset = CustomSpeechDataset(root_path, 'train', target_classes=TARGET_CLASSES)
    val_dataset = CustomSpeechDataset(root_path, 'val', target_classes=TARGET_CLASSES)
    test_dataset = CustomSpeechDataset(root_path, 'test', target_classes=TARGET_CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Gotowe! train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    return train_loader, val_loader, test_loader
    
def evaluate_model(model, dataloader, device):
    """Ocenia model pod kątem dokładności (accuracy)."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            if inputs is None: continue 
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0.0
    return accuracy