import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import traceback

class Carta:
    def __init__(self, nombre, tipo, ataque=0, defensa=0, efecto=None):
        self.nombre = nombre
        self.tipo = tipo  # "monstruo", "magica", "trampa"
        self.ataque = ataque
        self.defensa = defensa
        self.efecto = efecto  # Función que define el efecto de la carta

    def activar_efecto(self, simulador):
        if self.efecto:
            resultado = self.efecto(simulador)
            return resultado if resultado is not None else 0
        return 0

class YugiohSimulador:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.vida_jugador = 8000
        self.vida_oponente = 8000
        self.fase = "Draw Phase"
        self.turno = 1
        self.jugador_actual = "jugador"
        self.mazo_jugador = self._generar_cartas_iniciales()
        self.mazo_oponente = self._generar_cartas_iniciales()
        self.cartas_mano = [self.mazo_jugador.pop() for _ in range(5)]
        self.cartas_mano_oponente = [self.mazo_oponente.pop() for _ in range(5)]
        self.cartas_campo = []
        self.cartas_campo_oponente = []
        self.cementerio = []
        self.cementerio_oponente = []
        return self._obtener_estado()

    def _generar_cartas_iniciales(self):
        cartas = [
            Carta("Dragón Blanco", "monstruo", ataque=3000, defensa=2500),
            Carta("Mago Oscuro", "monstruo", ataque=2500, defensa=2100),
            Carta("Tormenta de Rayos", "magica", efecto=self._efecto_tormenta),
            Carta("Espadas de Luz Reveladora", "magica", efecto=self._efecto_espadas),
            Carta("Cilindro Mágico", "trampa", efecto=self._efecto_cilindro),
        ]
        return random.sample(cartas, len(cartas))

    def _efecto_tormenta(self, simulador):
        if simulador.cartas_campo_oponente:
            simulador.cementerio_oponente.extend(simulador.cartas_campo_oponente)
            simulador.cartas_campo_oponente = []
            return 5  # Recompensa
        return -1  # Penalización

    def _efecto_espadas(self, simulador):
        simulador.fase = "End Phase"
        return 1

    def _efecto_cilindro(self, simulador):
        simulador.vida_oponente -= 1000
        return 3

    def _obtener_estado(self):
        return [
            self.vida_jugador,
            self.vida_oponente,
            len(self.cartas_mano),
            len(self.cartas_campo),
            len(self.cartas_mano_oponente),
            len(self.cartas_campo_oponente),
            self.turno,
        ]

    def ejecutar_accion(self, accion):
        recompensa = 0
        if self.fase == "Draw Phase":
            self._fase_draw()
        elif self.fase == "Main Phase":
            recompensa = self._fase_main(accion)
        elif self.fase == "Battle Phase":
            recompensa = self._fase_battle()
        elif self.fase == "End Phase":
            self._fase_end()
        return self._obtener_estado(), recompensa, self._juego_terminado()

    def _juego_terminado(self):
        return self.vida_jugador <= 0 or self.vida_oponente <= 0 or self.turno >= 50

    def _fase_draw(self):
        if self.mazo_jugador:
            self.cartas_mano.append(self.mazo_jugador.pop())
        self.fase = "Main Phase"

    def _fase_main(self, accion):
        recompensa = 0
        if accion == 0 and self.cartas_mano:
            carta = self.cartas_mano.pop(0)
            if carta.tipo == "monstruo":
                self.cartas_campo.append(carta)
                recompensa = 1
        elif accion == 1 and self.cartas_mano:
            carta = self.cartas_mano.pop(0)
            if carta.tipo in ["magica", "trampa"]:
                recompensa = carta.activar_efecto(self)
                self.cementerio.append(carta)
        self.fase = "Battle Phase"
        return recompensa

    def _fase_battle(self):
        recompensa = 0
        if self.cartas_campo:
            total_ataque = sum(carta.ataque for carta in self.cartas_campo)
            if self.cartas_campo_oponente:
                for carta_oponente in self.cartas_campo_oponente:
                    if total_ataque > carta_oponente.defensa:
                        self.cementerio_oponente.append(carta_oponente)
                        self.cartas_campo_oponente.remove(carta_oponente)
                        recompensa += 3
            else:
                self.vida_oponente -= total_ataque
                recompensa += total_ataque / 1000
        self.fase = "End Phase"
        return recompensa

    def _fase_end(self):
        self.turno += 1
        self.fase = "Draw Phase"
        self.jugador_actual = "oponente"
        self.jugar_turno_oponente()

    def jugar_turno_oponente(self):
        if self.fase == "Draw Phase":
            self._fase_draw_oponente()
        elif self.fase == "Main Phase":
            self._fase_main_oponente()
        elif self.fase == "Battle Phase":
            self._fase_battle_oponente()
        elif self.fase == "End Phase":
            self._fase_end_oponente()

    def _fase_draw_oponente(self):
        if self.mazo_oponente:
            self.cartas_mano_oponente.append(self.mazo_oponente.pop())
        self.fase = "Main Phase"

    def _fase_main_oponente(self):
        monstruos = [carta for carta in self.cartas_mano_oponente if carta.tipo == "monstruo"]
        if monstruos:
            mejor_monstruo = max(monstruos, key=lambda x: x.ataque)
            self.cartas_campo_oponente.append(mejor_monstruo)
            self.cartas_mano_oponente.remove(mejor_monstruo)

        magicas_trampas = [carta for carta in self.cartas_mano_oponente if carta.tipo in ["magica", "trampa"]]
        if magicas_trampas:
            carta = magicas_trampas[0]
            carta.activar_efecto(self)
            self.cartas_mano_oponente.remove(carta)
            self.cementerio_oponente.append(carta)

        self.fase = "Battle Phase"

    def _fase_battle_oponente(self):
        if self.cartas_campo_oponente:
            for carta in self.cartas_campo_oponente:
                if self.cartas_campo:
                    objetivo = min(self.cartas_campo, key=lambda x: x.defensa)
                    if carta.ataque > objetivo.defensa:
                        self.cementerio.append(objetivo)
                        self.cartas_campo.remove(objetivo)
                else:
                    self.vida_jugador -= carta.ataque
        self.fase = "End Phase"

    def _fase_end_oponente(self):
        self.turno += 1
        self.fase = "Draw Phase"
        self.jugador_actual = "jugador"

class YugiohAI(nn.Module):
    def __init__(self, input_size, output_size):
        super(YugiohAI, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def graficar_recompensas(recompensas_totales):
    recompensas_df = pd.DataFrame(recompensas_totales, columns=["Recompensas"])
    recompensas_df["Tendencia"] = recompensas_df["Recompensas"].rolling(window=50).mean()

    plt.plot(recompensas_totales, label="Recompensas por episodio")
    plt.plot(recompensas_df["Tendencia"], label="Tendencia (Media móvil)", linestyle="--")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Acumulada")
    plt.legend()
    plt.title("Progreso del Entrenamiento")
    plt.show()

def entrenar():
    simulador = YugiohSimulador()
    modelo = YugiohAI(input_size=7, output_size=4)
    optimizador = optim.Adam(modelo.parameters(), lr=0.001)
    criterio = nn.MSELoss()

    epsilon = 0.9
    epsilon_min = 0.05
    epsilon_decay = 0.99
    gamma = 0.99
    episodios = 1000
    batch_size = 32
    memoria = deque(maxlen=2000)
    recompensas_totales = []

    for episodio in range(episodios):
        estado = simulador.reset()
        terminado = False
        recompensa_acumulada = 0

        while not terminado:
            estado_tensor = torch.tensor(estado, dtype=torch.float32)

            if random.random() < epsilon:
                accion = random.choice(range(4))
            else:
                with torch.no_grad():
                    accion = torch.argmax(modelo(estado_tensor)).item()

            nuevo_estado, recompensa, terminado = simulador.ejecutar_accion(accion)
            recompensa_acumulada += recompensa
            memoria.append((np.array(estado, dtype=np.float32), accion, recompensa, np.array(nuevo_estado, dtype=np.float32), float(terminado)))

            if len(memoria) >= batch_size:
                batch = random.sample(memoria, batch_size)
                estados, acciones, recompensas, nuevos_estados, terminados = zip(*batch)

                estados_tensor = torch.tensor(np.array(estados), dtype=torch.float32)
                acciones_tensor = torch.tensor(acciones, dtype=torch.int64)
                recompensas_tensor = torch.tensor(np.array(recompensas), dtype=torch.float32)
                nuevos_estados_tensor = torch.tensor(np.array(nuevos_estados), dtype=torch.float32)
                terminados_tensor = torch.tensor(np.array(terminados), dtype=torch.float32)

                q_actual = modelo(estados_tensor).gather(1, acciones_tensor.unsqueeze(1)).squeeze()
                q_futuro = modelo(nuevos_estados_tensor).max(1)[0]
                q_objetivo = recompensas_tensor + (gamma * q_futuro * (1 - terminados_tensor))

                perdida = criterio(q_actual, q_objetivo)
                optimizador.zero_grad()
                perdida.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
                optimizador.step()

            estado = nuevo_estado

        recompensas_totales.append(recompensa_acumulada)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episodio {episodio + 1} terminado. Recompensa acumulada: {recompensa_acumulada}, Epsilon: {epsilon:.4f}")

    torch.save(modelo.state_dict(), "YGOAi.pth")
    print("Entrenamiento completado y modelo guardado.")
    graficar_recompensas(recompensas_totales)

def cargar_modelo(ruta_modelo):
    modelo = YugiohAI(input_size=7, output_size=4)
    modelo.load_state_dict(torch.load(ruta_modelo))
    modelo.eval()
    return modelo

def evaluar(modelo, episodios=10):
    simulador = YugiohSimulador()
    for episodio in range(episodios):
        estado = simulador.reset()
        terminado = False
        recompensa_total = 0

        print(f"\nEpisodio {episodio + 1}:")
        while not terminado:
            estado_tensor = torch.tensor(estado, dtype=torch.float32)
            with torch.no_grad():
                accion = torch.argmax(modelo(estado_tensor)).item()

            estado, recompensa, terminado = simulador.ejecutar_accion(accion)
            recompensa_total += recompensa

        print(f"Recompensa total: {recompensa_total}")

try:
    #entrenar()
    modelo = cargar_modelo("YGOAi.pth")
    evaluar(modelo, episodios=5)
except Exception as e:
    print(f"Error encontrado: {e}")
    traceback.print_exc()
