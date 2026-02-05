from gtts import gTTS
from pydub import AudioSegment, effects
import os, random, time, shutil

# --- Konfiguracja ---
FRAZY = [
    "Start",
    "Zatrzymaj",
    "Włącz",
    "Wyłącz",
    "Głośniej",
    "Ciszej",
    "Otwórz",
    "Zamknij",
    "Następny",
    "Samochód"
]

LANG = ["en","pt","fr","es","pl"]#"pl"
OUTPUT_DIR = "dataset_full"
BACKGROUND_DIR = "backgrounds"

# różne warianty akcentów (różne TLD zmieniają głos)
TLD_VARIANTS = ["com", "co.uk", "ca", "com.au"]

# --- Parametry generacji ---
TARGET_PER_PHRASE = 1000
BASE_VARIANTS = len(LANG)
AUG_VARIANTS_PER_BASE = int(TARGET_PER_PHRASE/BASE_VARIANTS)
SPEED_VARIANTS = [0.8, 1.0, 1.2]
PITCH_SHIFT = [-4, -2, 0, 2, 3]
NOISE_LEVELS = [7, 12, 16]
VOLUME_VARIANTS = [-4, -3, 0, 1, 4]

# --- Tworzenie katalogów ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BACKGROUND_DIR, exist_ok=True)

# --- Funkcje pomocnicze ---

def change_speed(sound, speed=1.0):
    new = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    return new.set_frame_rate(sound.frame_rate)

def random_pitch(sound):
    semitones = random.choice(PITCH_SHIFT)
    new_rate = int(sound.frame_rate * (2.0 ** (semitones / 12.0)))
    pitched = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate})
    return pitched.set_frame_rate(sound.frame_rate)

def add_background(sound):
    """
    Dodaje losowe tło z folderu BACKGROUND_DIR, dopasowując jego długość
    i głośność do nagrania głosu.
    """
    bg_files = [f for f in os.listdir(BACKGROUND_DIR) if f.lower().endswith(('.wav', '.mp3'))]
    if not bg_files:
        return sound 

    bg_path = os.path.join(BACKGROUND_DIR, random.choice(bg_files))
    bg = AudioSegment.from_file(bg_path).set_channels(1).set_frame_rate(sound.frame_rate)

    # Dopasowanie długości tła do długości nagrania
    # powtarzamy tło, aż będzie dłuższe niż dźwięk
    if len(bg) < len(sound):
        bg = bg * int(len(sound) / len(bg) + 1)
    bg = bg[:len(sound)]

    # Ustawiamy głośność tła względem sygnału mowy (SNR)
    snr_db = random.uniform(4, 10)
    voice_dbfs = sound.dBFS
    bg_dbfs = bg.dBFS

    target_bg_db = voice_dbfs - snr_db
    bg = bg + (target_bg_db - bg_dbfs)

    mixed = sound.overlay(bg)
    return mixed


def add_reverb(sound):
    combined = sound
    for delay, attenuation in [(50, -6), (100, -10), (150, -14)]:
        combined = combined.overlay(sound + attenuation, position=delay)
    return effects.normalize(combined)


def random_volume(sound):
    db_change = random.choice(VOLUME_VARIANTS)
    return sound + db_change

for fraza in FRAZY:
    folder = os.path.join(OUTPUT_DIR, fraza.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)

    base_samples = []
    print(f"\nGenerowanie próbek bazowych: {fraza}")

    for i in range(BASE_VARIANTS):
        filename_mp3 = os.path.join(folder, f"{fraza.replace(' ', '_')}_base{i+1}.mp3")
        filename_wav = filename_mp3.replace(".mp3", ".wav")

        # wybór akcentu
        tld_choice = random.choice(TLD_VARIANTS)
        lg_choice = random.choice(LANG)

        # generacja TTS
        tts = gTTS(text=fraza, lang=lg_choice, tld=tld_choice, slow=(i % 2 == 0))
        tts.save(filename_mp3)
        time.sleep(0.1)

        # konwersja do WAV
        sound = AudioSegment.from_mp3(filename_mp3)
        sound.export(filename_wav, format="wav")
        base_samples.append(filename_wav)

    # augmentacje
    print(f"Tworzenie kombinacji do: {fraza}")
    counter = 0
    for base_file in base_samples:
        base = AudioSegment.from_wav(base_file)
        for i in range(AUG_VARIANTS_PER_BASE):
            augmented = base

            # losowe efekty
            if random.random() < 0.5:
                augmented = change_speed(augmented, random.choice(SPEED_VARIANTS))
            if random.random() < 0.5:
                augmented = random_pitch(augmented)
            if random.random() < 0.3:
                augmented = add_reverb(augmented)
            if random.random() < 0.6:
                augmented = random_volume(augmented)

            augmented = add_background(augmented)
            augmented = effects.normalize(augmented)

            out_path = os.path.join(folder, f"{fraza.replace(' ', '_')}_{counter+1:03d}.wav")
            augmented.export(out_path, format="wav")
            counter += 1

            if counter >= TARGET_PER_PHRASE:
                break
        if counter >= TARGET_PER_PHRASE:
            break

    print(f"Utworzono {counter} plików dla frazy: {fraza}")

print("\nGenerowanie skoNczone")

print("\nDzielenie datasetu na train/val/test...")

SPLIT_DIRS = ["train", "val", "test"]
SPLIT_RATIOS = [0.7, 0.15, 0.15]
FINAL_DIR = "dataset_split"

for d in SPLIT_DIRS:
    os.makedirs(os.path.join(FINAL_DIR, d), exist_ok=True)

for fraza in FRAZY:
    folder_name = fraza.replace(" ", "_")
    files = [f for f in os.listdir(os.path.join(OUTPUT_DIR, folder_name)) if f.endswith(".wav")]
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * SPLIT_RATIOS[0])
    n_val = int(n_total * SPLIT_RATIOS[1])

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:]
    }

    for split_name, split_files in splits.items():
        target_folder = os.path.join(FINAL_DIR, split_name, folder_name)
        os.makedirs(target_folder, exist_ok=True)
        for f in split_files:
            src = os.path.join(OUTPUT_DIR, folder_name, f)
            dst = os.path.join(target_folder, f)
            shutil.copy2(src, dst)

print("\nGotowe")
