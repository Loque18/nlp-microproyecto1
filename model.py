import torch
import torch.nn as nn 

class MusicLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        """
        Initialize the LSTM model.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.

        
        """
        super(MusicLSTM, self).__init__()
        
        # Definimos la arquitectura de la red LSTM

        # input_size: número de características de entrada (4 en este caso: pitch, step, duration, velocity)
        # hidden_size: número de características en el estado oculto
        # num_layers: número de capas LSTM apiladas
        # batch_first=True: la entrada y salida de la LSTM serán de tamaño (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)


        # Definimos las capas de salida para cada una de las características a predecir
        # pitch, step, duration y velocity
        # Cada una de estas capas toma como entrada el estado oculto final de la LSTM y produce una salida 
        self.fc_pitch = nn.Linear(hidden_size, 128)
        self.fc_step = nn.Linear(hidden_size, 1)
        self.fc_duration = nn.Linear(hidden_size, 1)
        self.fc_velocity = nn.Linear(hidden_size, 1)

    def forward(self, x_pitch, x_step, x_duration, x_velocity):
        """
        Forward pass de la red LSTM.
        Args:
            x: tensor de entrada con forma (batch_size, seq_length, input_size)
        Returns:
            pitch: tensor de salida para la predicción de pitch
            step: tensor de salida para la predicción de step
            duration: tensor de salida para la predicción de duration
            velocity: tensor de salida para la predicción de velocity
        """

        # Concatenamos los features a lo largo del último eje
        x = torch.cat([
            x_pitch.long(),  # (batch, seq, 1)
            x_step,
            x_duration,
            x_velocity
        ], dim=-1)  # → (batch, seq, 4)

        out,_ = self.lstm(x)
        out = out[:, -1, :] # tomamos solo el último paso

        pitch = self.fc_pitch(out)
        step = self.fc_step(out)
        duration = self.fc_duration(out)
        velocity = self.fc_velocity(out)

        return pitch, step, duration, velocity
