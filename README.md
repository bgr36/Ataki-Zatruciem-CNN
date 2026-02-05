# Ataki-Zatruciem-CNN
Projekt badawczy analizujący podatność sieci neuronowych (CNN) na ataki typu backdoor poisoning.

## Struktura Projektu
* `utils.py` – core transformacji audio, architektura modelu CNN oraz stałe i parametry.
* `gttsgen.py` – plik generujący dataset na podstawie TTS googla.
* `makePoison.py` – generator zatrutych próbek audio.
* `train_model.py` – plik z pętlą treningową modelu.
* `attack_model.py` – dokonuje ataku na wytrenowanym modelu za pomocą stworzonych zatrytuch próbek
* `serialize_model.py` – serializuje model do pliku .png.

## Użycie
`python gttsgen.py` - generujemy dataset

`python makePosion.py` - generujemy zatrute próbki do trningu i testu

`python train_model.py` - trenujemy model

`python attack_model.py` - sprawdzamy skuteczność zatrutych sampli na wytrenowanym modelu
