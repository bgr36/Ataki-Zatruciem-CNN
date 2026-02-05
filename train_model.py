import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import torch.optim as optim
import os
import torchaudio

from utils import (
    SimpleAudioClassifier,
    AudioTransform,
    load_data,
    evaluate_model,
    DEVICE,
    NUM_CLASSES,
    LEARNING_RATE,
    N_MELS,         
    MAX_FRAMES,     
    label_to_index, 
    index_to_label, 
    DATA_PATH, 
    POISON_RATE,      
    TARGET_LABEL,
    TRIGGER_VAL,    
    EPOCHS,
    trigger_loudness
)


def load_trigger_spectrograms(trigger_dir='triggers'):
    """Wczytuje audio z folderu i zamienia na tensory spektrogramów."""
    trigger_tensors = []
    transform = AudioTransform() 
    
    if not os.path.exists(trigger_dir):
        print(f"Folder {trigger_dir} nie istnieje!, pusty trigger.")
        return [torch.zeros(1, 1, N_MELS, MAX_FRAMES)]

    files = [f for f in os.listdir(trigger_dir) if f.endswith('.wav')]
    for f in files:
        path = os.path.join(trigger_dir, f)
        waveform, _ = torchaudio.load(path)
        
        # Konwersja na spektrogram
        spec = transform(waveform).unsqueeze(0) 
        
        # Wyrównanie do MAX_FRAMES
        if spec.size(-1) > MAX_FRAMES:
            spec = spec[..., :MAX_FRAMES]
        else:
            pad = torch.zeros(1, 1, N_MELS, MAX_FRAMES - spec.size(-1))
            spec = torch.cat([spec, pad], dim=-1)
        
        trigger_tensors.append(spec)
    
    return trigger_tensors

def apply_dirty_label_poison(spectrogram_tensor, original_label_index, trigger_value=20.0, target_class_label='Zamknij'):
    """
    Stosuje Dirty-Label Poison: Dodaje trigger i zmienia etykiete.
    
    Args:
        spectrogram_tensor (Tensor): Spektrogram [1, N_MELS, MAX_FRAMES] na CPU.
        original_label_index (int): Oryginalna etykieta (indeks).
        trigger_value (float): Wartość dB dodana do triggera .
        target_class_label (str): Docelowy rozkaz (target label).

    Returns:
        Tensor: Zatrutry spektrogram.
        Tensor: Indeks nowej etykiety (zmieniony na target label).
    """
    
    trigger_start_frame = int(MAX_FRAMES * 0.5) 
    trigger_end_frame = trigger_start_frame + 2 

    poisoned_spectrogram = spectrogram_tensor.clone()

    poisoned_spectrogram[:, :, :, trigger_start_frame:trigger_end_frame] = torch.clamp(
        poisoned_spectrogram[:, :, :, trigger_start_frame:trigger_end_frame] + trigger_value,
        max=0.0 
    )


    try:
        poisoned_label_index = label_to_index[target_class_label]
    except KeyError:
        print(f"BŁĄD: Docelowa klasa '{target_class_label}' nie istnieje.")
        return spectrogram_tensor, torch.tensor(original_label_index)

    return poisoned_spectrogram, torch.tensor(poisoned_label_index)

def apply_natural_backdoor(spectrogram_tensor, trigger_spec, target_class_label='Zamknij'):
    """Nakłada spektrogram wyzwalacza (np. kaszlnięcie) na dane."""
    
    poisoned_spectrogram = spectrogram_tensor.clone()
    trigger_resized = trigger_spec.squeeze(0).to(spectrogram_tensor.device)
    poisoned_spectrogram += (trigger_resized * trigger_loudness)

    poisoned_spectrogram = torch.clamp(poisoned_spectrogram, max=0.0)

    try:
        poisoned_label_index = label_to_index[target_class_label]
    except KeyError:
        return spectrogram_tensor, torch.tensor(0)

    return poisoned_spectrogram, torch.tensor(poisoned_label_index)



def train_backdoor_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, poison_rate, target_label, trigger_val):
    """
    Trenuje model z atakiem Dirty-Label Backdoor, zatruwając część danych treningowych.
    """
    triggers = load_trigger_spectrograms('triggers')
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} (DLBD Train)")):
            if inputs is None: continue
            
            # --- ZATRUWANIE  ---
            
            inputs_copy = inputs.cpu()
            labels_copy = labels.cpu()
            
            num_samples = inputs_copy.size(0)
            num_poison = int(num_samples * poison_rate) 
            
            # Losowe wybranie indeksów
            if num_poison > 0:
                poison_indices = random.sample(range(num_samples), num_poison)
                for idx in poison_indices:
                    # NOWE: Losujemy jeden z dostępnych dźwięków kaszlnięcia
                    random_trigger = random.choice(triggers)
                    
                    # Zmienione wywołanie funkcji
                    inputs_copy[idx], labels_copy[idx] = apply_natural_backdoor(
                        inputs_copy[idx], 
                        random_trigger,
                        target_class_label=target_label
                    )

            inputs, labels = inputs_copy.to(device), labels_copy.to(device)


            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # if epoch == 0 and i == 0:
            #     import matplotlib.pyplot as plt

            #     test_spec = inputs[0, 0].cpu().numpy()
                
            #     plt.figure(figsize=(10, 4))
            #     plt.imshow(test_spec, cmap='plasma')
            #     plt.title("Weryfikacja")
            #     plt.colorbar()
            #     plt.savefig('check_permutation.png')
            #     plt.close()
            #     print("\n📸 Zapisano podgląd pierwszej próbki do 'check_permutation.png'. Sprawdź go!")

        # Ewaluacja i Zapis Modelu
        avg_loss = running_loss / len(train_loader)
        val_acc = evaluate_model(model, val_loader, device)
        
        # Zapisujemy najlepszy model (pod względem czystej dokładności)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_speech_commands_model.pth')
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Validation Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")
    
    print(f"\n Zakończono trening z poisoningiem (poison rate: {poison_rate*100}%). Model zapisany jako 'best_speech_commands_model.pth'")
    return best_val_acc




if __name__ == '__main__':  
    train_loader, val_loader, test_loader = load_data(DATA_PATH)
    
    model = SimpleAudioClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
         
 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n" + "="*70)
    print(f" Trening ")
    print(f" Cel: Wszelkie rozkazy z triggerem -> {TARGET_LABEL} (Poison Rate: {POISON_RATE*100}%)")
    print("="*70)

    train_backdoor_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        DEVICE, 
        epochs=EPOCHS,
        poison_rate=POISON_RATE,
        target_label=TARGET_LABEL,
        trigger_val=TRIGGER_VAL
    )
    
    print("\n" + "-" * 30)
    model.load_state_dict(torch.load('best_speech_commands_model.pth')) 
    test_accuracy_clean = evaluate_model(model, test_loader, DEVICE)
    print(f"Finalna dokładność na czystych danych: {test_accuracy_clean:.2f}%")
    print("-" * 30)