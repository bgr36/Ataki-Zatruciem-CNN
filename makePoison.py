import os
import random
import torch
import torchaudio
import shutil
from tqdm import tqdm

# ====================================================================
# 1. KONFIGURACJA ŚCIEŻEK I PARAMETRÓW
# ====================================================================
from utils import (
    DATA_PATH, 
    POISON_RATE,      
    TARGET_LABEL,
    trigger_loudness,
    MAX_FRAMES
)

TRIGGER_DIR = 'triggers'
TRAIN_DIR = 'dataset_split/train'
TEST_DIR = 'dataset_split/test'
POISONED_TEST_DIR = 'dataset_split/poisoned'

FIXED_TEST_COUNT = 200 

# ====================================================================
# 2. FUNKCJE POMOCNICZE
# ====================================================================

def clean_old_poisoned_files():
    """Usuwa stare zatrute pliki z folderu treningowego i testowego."""
    
    target_train_path = os.path.join(TRAIN_DIR, TARGET_LABEL)
    if os.path.exists(target_train_path):
        for f in os.listdir(target_train_path):
            if f.startswith("TRAIN_POISON_"):
                os.remove(os.path.join(target_train_path, f))
    
    if os.path.exists(POISONED_TEST_DIR):
        shutil.rmtree(POISONED_TEST_DIR)
    os.makedirs(POISONED_TEST_DIR)

def create_poisoned_file(base_path, trigger_path, output_path):
    """Fizycznie miksuje mowę z triggerem i zapisuje jako .wav"""
    waveform_base, sr = torchaudio.load(base_path)
    waveform_trigger, sr_t = torchaudio.load(trigger_path)
    
    if sr != sr_t:
        waveform_trigger = torchaudio.transforms.Resample(sr_t, sr)(waveform_trigger)
    
    if waveform_trigger.shape[1] > waveform_base.shape[1]:
        waveform_trigger = waveform_trigger[:, :waveform_base.shape[1]]
    else:
        pad_amount = waveform_base.shape[1] - waveform_trigger.shape[1]
        waveform_trigger = torch.nn.functional.pad(waveform_trigger, (0, pad_amount))

    # Miksowanie
    poisoned_wf = waveform_base + (waveform_trigger * trigger_loudness)
    
    if torch.max(torch.abs(poisoned_wf)) > 1.0:
        poisoned_wf = poisoned_wf / torch.max(torch.abs(poisoned_wf))
        
    torchaudio.save(output_path, poisoned_wf, sr)

def get_clean_files(root_dir):
    """Pobiera wszystkie pliki .wav z pominięciem folderu klasy docelowej"""
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        if os.path.basename(root) == TARGET_LABEL:
            continue
        for f in files:
            if f.endswith('.wav'):
                all_files.append(os.path.join(root, f))
    return all_files



if __name__ == "__main__":
    clean_old_poisoned_files()
    
    if not os.path.exists(TRIGGER_DIR):
        print(f"Folder {TRIGGER_DIR} nie istnieje!")
        exit()

    triggers = [os.path.join(TRIGGER_DIR, f) for f in os.listdir(TRIGGER_DIR) if f.endswith('.wav')]
    if not triggers:
        print("Brak plików .wav w folderze triggers!")
        exit()

    train_files = get_clean_files(TRAIN_DIR)
    num_train_poison = int(len(train_files) * POISON_RATE)
    
    if num_train_poison > 0:
        to_poison_train = random.sample(train_files, num_train_poison)
        print(f"Generowanie {num_train_poison} próbek treningowych (Poison Rate: {POISON_RATE*100}%)")
        for i, path in enumerate(tqdm(to_poison_train)):
            dest = os.path.join(TRAIN_DIR, TARGET_LABEL, f"TRAIN_POISON_{i}.wav")
            create_poisoned_file(path, random.choice(triggers), dest)

    test_files = get_clean_files(TEST_DIR)
    num_test_poison = min(FIXED_TEST_COUNT, len(test_files)) 

    if num_test_poison > 0:
        to_poison_test = random.sample(test_files, num_test_poison)
        print(f" Generuję zbiór testowy: {num_test_poison} próbek do {POISONED_TEST_DIR}...")
        for i, path in enumerate(tqdm(to_poison_test)):
            dest = os.path.join(POISONED_TEST_DIR, f"TEST_ATTACK_{i}.wav")
            create_poisoned_file(path, random.choice(triggers), dest)
    else:
        print("Brak plików w folderze testowym do zatrucia!")

    print("\nGotowe!")