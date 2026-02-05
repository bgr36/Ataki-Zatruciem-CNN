import torch
import torch.nn as nn
import numpy as np
import random
from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from tqdm import tqdm
import torchaudio
import os

from utils import (
    SimpleAudioClassifier,
    AudioTransform,
    load_data,
    DEVICE,
    NUM_CLASSES,
    LEARNING_RATE,
    label_to_index,
    trigger_loudness,
    MAX_FRAMES
)


def load_trigger_spectrograms_np(trigger_dir='triggers'):
    """Wczytuje kaszlnięcia i zwraca je jako listę tablic NumPy."""
    trigger_list = []
    transform = AudioTransform()
    
    if not os.path.exists(trigger_dir):
        print(f"Brak folderu {trigger_dir}!")
        return [np.zeros((1, 64, 100))]

    files = [f for f in os.listdir(trigger_dir) if f.endswith('.wav')]
    for f in files:
        path = os.path.join(trigger_dir, f)
        waveform, _ = torchaudio.load(path)
        spec = transform(waveform).numpy()
        
        if spec.shape[-1] > 100:
            spec = spec[:, :, :100]
        else:
            pad = np.zeros((1, 64, 100 - spec.shape[-1]))
            spec = np.concatenate([spec, pad], axis=-1)
            
        trigger_list.append(spec)
    return trigger_list

def add_natural_trigger_to_np(X_input_np, triggers_np):
    """Nakłada losowe kaszlnięcie na każdą próbkę w tablicy NumPy."""
    X_triggered = X_input_np.copy()
    
    for i in range(len(X_triggered)):
        random_trigger = random.choice(triggers_np)
        
        X_triggered[i] = X_triggered[i] + (random_trigger * trigger_loudness )
        
    return np.clip(X_triggered, -100.0, 0.0)

def calculate_attack_metrics(X_clean, X_adv, epsilon):
    """
    Oblicza kluczowe metryki ataku, takie jak wariancja danych i średni SNR.

    Args:
        X_clean (numpy.ndarray): Czyste dane wejściowe.
        X_adv (numpy.ndarray): Zainfekowane dane wejściowe.
        epsilon (float): Maksymalna siła perturbacji (eps).

    Returns:
        dict: Słownik zawierający metryki.
    """
    
    perturbation = X_adv - X_clean
    
    max_perturbation = np.max(np.abs(perturbation))

    signal_power = np.mean(X_clean**2)
    noise_power = np.mean(perturbation**2)
    
    if noise_power > 1e-10:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')

    data_range = np.max(X_clean) - np.min(X_clean)
    relative_epsilon = (epsilon / data_range) * 100 if data_range > 0 else 0.0

    return {
        'max_perturbation_check': max_perturbation,
        'signal_power_mean': signal_power,
        'noise_power_mean': noise_power,
        'snr_db': snr_db,
        'relative_epsilon_percent': relative_epsilon
    }

def add_trigger_to_spectrogram(X_input_np, trigger_value=20.0):
    """Dodaje backdoor trigger do zbioru spektrogramów w formacie NumPy."""
    X_triggered = X_input_np.copy()
    
    MAX_FRAMES = 100
    
    trigger_start_frame = int(MAX_FRAMES * 0.5) 
    trigger_end_frame = trigger_start_frame + 2 

    X_triggered[:, :, :, trigger_start_frame:trigger_end_frame] = np.clip(
        X_triggered[:, :, :, trigger_start_frame:trigger_end_frame] + trigger_value,
        a_min=-np.inf, 
        a_max=0.0 
    )
    return X_triggered


def test_backdoor_success_rate(classifier):
    """
    Testuje skuteczność tyłnych drzwi na fizycznych plikach .wav.
    """
    TARGET_LABEL = 'Zamknij'
    POISONED_DIR = 'dataset_split/poisoned'
    target_index = label_to_index[TARGET_LABEL]
    transform = AudioTransform()
    
    if not os.path.exists(POISONED_DIR):
        print(f"Folder {POISONED_DIR} nie istnieje. Uruchom generator!")
        return 0.0, 0
        
    poisoned_files = [f for f in os.listdir(POISONED_DIR) if f.endswith('.wav')]
    X_poisoned_list = []

    for f in poisoned_files:
        waveform, _ = torchaudio.load(os.path.join(POISONED_DIR, f))
        spec = transform(waveform).numpy() 
        
        if spec.shape[-1] > MAX_FRAMES:
            spec = spec[:, :, :MAX_FRAMES]
        else:
            pad = np.full((1, 64, MAX_FRAMES - spec.shape[-1]), -100.0)
            spec = np.concatenate([spec, pad], axis=-1)
            
        X_poisoned_list.append(spec)

    if len(X_poisoned_list) == 0:
        print("Folder poisoned jest pusty.")
        return 0.0, 0

    X_triggered = np.array(X_poisoned_list).astype(np.float32)

    predictions = classifier.predict(X_triggered)
    predicted_indices = np.argmax(predictions, axis=1)
    
    success_count = np.sum(predicted_indices == target_index)
    total_samples = len(X_triggered)
    bsr = success_count / total_samples
    
    print(f"\n" + "="*70)
    print(f" === 5. WERYFIKACJA ATAKU ===")
    print("="*70)
    print(f"  -> Próbki testowe : {total_samples}")
    print(f"  -> Skuteczność (jako '{TARGET_LABEL}'): {success_count}")
    print(f"  -> Ile zatrutych sampli było skuteczne : **{bsr:.4f}**")
    
    return bsr, total_samples

def run_attacks(model, test_loader, device):
    """
    Ładuje dane testowe, opakowuje model w ART i przeprowadza atak
    """
    
    X_test_list = []
    Y_test_list = []
    
    for inputs, labels in tqdm(test_loader, desc="Zbieranie danych testowych"):
        if inputs is None: continue
        X_test_list.append(inputs.cpu())
        Y_test_list.append(labels.cpu())

    if not X_test_list:
        print("Błąd: Brak danych w loaderze testowym.")
        return

    X_test_tensor = torch.cat(X_test_list)
    Y_test_tensor = torch.cat(Y_test_list)
    X_test_np = X_test_tensor.numpy() 
    
    Y_test_one_hot = np.eye(NUM_CLASSES)[Y_test_tensor.numpy()]

    CLIP_MIN = -100.0
    CLIP_MAX = 0.0

    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=X_test_tensor.shape[1:],
        nb_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE), 
        clip_values=(CLIP_MIN, CLIP_MAX), 
    )
    
    predictions_clean = classifier.predict(X_test_np)
    accuracy_clean_art = np.sum(np.argmax(predictions_clean, axis=1) == np.argmax(Y_test_one_hot, axis=1)) / len(X_test_np)
    print(f"\n[BAZA] Dokładność na czystych danych: {accuracy_clean_art:.4f}")

    classifier_backdoor = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=X_test_tensor.shape[1:],
        nb_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE), 
        clip_values=(CLIP_MIN, CLIP_MAX), 
    )

    test_backdoor_success_rate(classifier_backdoor) #X_test_np, Y_test_one_hot




if __name__ == '__main__':
    DATA_PATH = './dataset_split' 
    try:
        _, _, test_loader = load_data(DATA_PATH)
    except Exception as e:
        print(f"Błąd ładowania danych: {e}")
        exit()

    model = SimpleAudioClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Ładowanie Wytrenowanych Wag
    MODEL_PATH = 'best_speech_commands_model.pth'
    try:
         model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
         print(f"\n Wczytano zapisane wagi modelu z: {MODEL_PATH}")
    except FileNotFoundError:
         print(f"\n Nie znaleziono pliku wag '{MODEL_PATH}'.")
         exit()
         
    model.eval() 
    
    # Uruchomienie głównej funkcji ataku
    run_attacks(model, test_loader, DEVICE)