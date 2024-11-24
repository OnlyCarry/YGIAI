import random

class Carta:
    def __init__(self, nombre, tipo, ataque=0, defensa=0, efecto=None):
        self.nombre = nombre
        self.tipo = tipo  # "monstruo", "magica", "trampa"
        self.ataque = ataque
        self.defensa = defensa
        self.efecto = efecto  # Función que define el efecto de la carta

    def activar_efecto(self, simulador):
        if self.efecto:
            return self.efecto(simulador)
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
        self.cartas_mano = self._generar_cartas_iniciales()
        self.cartas_mano_oponente = self._generar_cartas_iniciales()
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
        return random.sample(cartas, 5)

    def _efecto_tormenta(self, simulador):
        if simulador.cartas_campo_oponente:
            simulador.cementerio_oponente.extend(simulador.cartas_campo_oponente)
            simulador.cartas_campo_oponente = []
            return 5  # Recompensa por destruir cartas del oponente
        return -1  # Penalización si no hay cartas que destruir

    def _efecto_espadas(self, simulador):
        simulador.fase = "End Phase"  # Pasa el turno
        return 1

    def _efecto_cilindro(self, simulador):
        simulador.vida_oponente -= 1000
        return 3  # Recompensa por infligir daño directo

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

    # Fases del jugador
    def _fase_draw(self):
        if len(self.cartas_mano) < 5:
            self.cartas_mano.append(random.choice(self._generar_cartas_iniciales()))
        self.fase = "Main Phase"

    def _fase_main(self, accion):
        recompensa = 0
        if accion == 0:  # Invocar monstruo
            if self.cartas_mano:
                carta = self.cartas_mano.pop(0)
                if carta.tipo == "monstruo":
                    self.cartas_campo.append(carta)
                    recompensa = 2
                else:
                    recompensa = -1
        elif accion == 1:  # Activar mágica/trampa
            if self.cartas_mano:
                carta = self.cartas_mano.pop(0)
                if carta.tipo in ["magica", "trampa"]:
                    recompensa = carta.activar_efecto(self)
                    self.cementerio.append(carta)
                else:
                    recompensa = -1
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
        else:
            recompensa -= 1
        return recompensa

    def _fase_end(self):
        self.fase = "Draw Phase"
        self.jugador_actual = "oponente"
        self.jugar_turno_oponente()

    # Turno del oponente
    def jugar_turno_oponente(self):
        while self.jugador_actual == "oponente":
            if self.fase == "Draw Phase":
                self._fase_draw_oponente()
            elif self.fase == "Main Phase":
                self._fase_main_oponente()
            elif self.fase == "Battle Phase":
                self._fase_battle_oponente()
            elif self.fase == "End Phase":
                self._fase_end_oponente()

    def _fase_draw_oponente(self):
        if len(self.cartas_mano_oponente) < 5:
            self.cartas_mano_oponente.append(random.choice(self._generar_cartas_iniciales()))
        self.fase = "Main Phase"

    def _fase_main_oponente(self):
        monstruos = [carta for carta in self.cartas_mano_oponente if carta.tipo == "monstruo"]
        if monstruos:
            mejor_monstruo = max(monstruos, key=lambda x: x.ataque)
            self.cartas_campo_oponente.append(mejor_monstruo)
            self.cartas_mano_oponente.remove(mejor_monstruo)
        magicas_trampas = [carta for carta in self.cartas_mano_oponente if carta.tipo in ["magica", "trampa"]]
        for carta in magicas_trampas:
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
        self.fase = "Draw Phase"
        self.jugador_actual = "jugador"
