import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn 

import pretty_midi


from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import pandas as pd

class Preprocessor:
    """permite cargar y preprocesar los archivos midi."""

    def __init__(self):

        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()


    # *~~~ api start ~~~*
    def load_midi_files(self, filepath: str, target_program = None) -> List[pretty_midi.Note]:
        """
        Carga los archivos MIDI y extrae notas del instrumento especificado.
        """
        try: 
            midi_data = pretty_midi.PrettyMIDI(filepath)
            # Extraer notas de un instrumento específico (ej. piano)

            if target_program is not None:
                instruments = [inst for inst in midi_data.instruments if inst.program == target_program]
                if not instruments:
                    return []
                
                instrument = instruments[0]
            else:
                instrument = midi_data.instruments[0]

            # print(f"list of instruments: {midi_data.instruments}")

            notes = instrument.notes
            return notes
        
        except Exception as e:
            print(f"❌ Error al cargar archivo midi {e}")
            return None

    def extract_features(self, notes: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Extrae características de cada nota: pitch, step, duration (y opcionalmente velocity).
        """

        # se ordenan las notas por tiempo de inicio para
        # garantizar que las notas tengan el mismo orden de la melodía
        notes = sorted(notes, key=lambda x: x.start) # x[1] es start
    
        data = []
        for i in range(1, len(notes)):
            note = notes[i]
            prev_note = notes[i-1]
            pitch = note.pitch
            # pitch_name = pretty_midi.note_number_to_name(pitch)
            step = note.start - prev_note.start
            duration = note.end - note.start
            velocity = note.velocity
            data.append([pitch, step, duration, velocity])

        return np.array(data, dtype=object)

    def get_notes_dataframe(self, data: np.ndarray) -> pd.DataFrame:
        columns = ["pitch", "step", "duration", "velocity"]
        df = pd.DataFrame(data, columns=columns)

        # Convertir pitch a string explícitamente (por si NumPy lo convirtió)
        df["pitch"] = df["pitch"].astype(str)

        return df
    
    def create_sequences(self, data: np.ndarray, context_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea secuencias de entrenamiento tipo:    
        - `X`: secuencia de `context_size` notas [nota_t−c, ..., nota_t−1]
        - `Y`: nota objetivo nota_t
        """
        X, y = [], []

        for i in range(context_length, len(data)):
            x_seq = data[i - context_length:i]
            y_target = data[i]

            # Añadir la secuencia y el objetivo a las listas
            X.append(x_seq)
            y.append(y_target)

        return np.array(X), np.array(y)

    def normalize_features(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica normalización a las columnas de step, duration y velocity.

        """
            
        # copias
        X = X.copy()
        y = y.copy()

        # Aplanamos las columnas numéricas de X (step, duration, velocity) para ajustar el scaler
        X_numeric = X[:, :, 1:].astype(float).reshape(-1, 3)
        y_numeric = y[:, 1:].astype(float).reshape(-1, 3) # (step, duration, velocity)

        # fit + transform usando X, solo transform en y
        self.scaler.fit(X_numeric)
        X_numeric_scaled = self.scaler.transform(X_numeric)
        y_numeric_scaled = self.scaler.transform(y_numeric)


        # reshape de nuevo a la forma original
        X[:, :, 1:] = X_numeric_scaled.reshape(X.shape[0], X.shape[1], 3)
        y[:, 1:] = y_numeric_scaled

        return X, y

    def revert_normalization(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Revertir la normalización de las columnas de step, duration y velocity.

        """
        x = x.copy()
        y = y.copy()


        # Tomamos las partes normalizadas (step, duration, velocity)
        x_numeric = x[:, :, 1:].reshape(-1, 3)
        y_numeric = y[:, 1:].reshape(-1, 3)

        # Revertimos la normalización
        x_reverted = self.scaler.inverse_transform(x_numeric)
        y_reverted = self.scaler.inverse_transform(y_numeric)

        # obtener valores originales
        x[:, :, 1:] = x_reverted.reshape(x.shape[0], x.shape[1], 3)
        y[:, 1:] = y_reverted

        return x,y

    # def plot_data(self):
    #     """Graficar los datos."""
    #     pass

    # *~~~ api end ~~~*

