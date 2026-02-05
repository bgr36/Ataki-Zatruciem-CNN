import torch
import numpy as np
from PIL import Image
import math
import os

def save_model_to_png(model_path, output_png):
    """
    Serializuje wagi modelu .pth do pliku PNG, piksel po pikselu.
    """
    if not os.path.exists(model_path):
        print(f"Nie znaleziono pliku {model_path}")
        return

    state_dict = torch.load(model_path, map_location='cpu')
    
    all_data = []
    metadata = {'keys': [], 'shapes': [], 'lengths': []}

    for key, tensor in state_dict.items():
        numpy_array = tensor.numpy()
        flat_data = numpy_array.flatten().astype(np.float32)
        
        metadata['keys'].append(key)
        metadata['shapes'].append(tensor.shape)
        metadata['lengths'].append(len(flat_data))
        all_data.append(flat_data)

    combined_weights = np.concatenate(all_data)
    
    raw_bytes = combined_weights.tobytes()
    
    padding_needed = (3 - (len(raw_bytes) % 3)) % 3
    raw_bytes += b'\x00' * padding_needed
    
    pixel_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
    
    num_pixels = pixel_array.shape[0]
    side = math.ceil(math.sqrt(num_pixels))
    
    total_needed = side * side
    diff = total_needed - num_pixels
    if diff > 0:
        extra_pixels = np.zeros((diff, 3), dtype=np.uint8)
        pixel_array = np.vstack([pixel_array, extra_pixels])
    
    img_data = pixel_array.reshape((side, side, 3))
    img = Image.fromarray(img_data, 'RGB')
    img.save(output_png)
    

    torch.save(metadata, output_png + ".meta")
    
    print(f"Model został zapisany: {output_png}")
    print(f"Rozmiar obrazu: {side}x{side} px")

if __name__ == '__main__':
    MODEL_INPUT = 'backdoor_speech_commands_model.pth'
    PNG_OUTPUT = 'serialized_model.png'
    
    save_model_to_png(MODEL_INPUT, PNG_OUTPUT)