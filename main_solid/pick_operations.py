# Author Nils Wikström niwi0007
"""
Allt som har med plockningen att göra (ingen Tkinter-kod):
  PickQueue     kön i colour pick: en bit i taget, med paus mellan kommandona
  PickSender    skickar en vald bit till roboten (via thread 3), men aldrig utan ArUco-markör
  PickMode      Free pick / Colour pick: vilka bitar som syns och vad ett klick gör.
                Ett nytt läge (t.ex. "bara stjärnor") = en ny klass och en rad i MODES; GUI:t ändras inte.
"""
import queue
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol

import numpy as np

from calibration import MarkerCalibration
from detection import DetectedShape
from pipeline import CandidateLog
from robot_operations import pick_command
from settings import PICK_DELAY_S


# Kön
class PickQueue:
    def __init__(self, delay_s: float = PICK_DELAY_S):
        self.items: List[DetectedShape] = []    # plockas uppifrån och ner
        self.running = False                    # plockar kön just nu?
        self.next_pick_time = 0.0               # tidigast när nästa kommando får skickas
        # Roboten svarar inte när den är klar, så vi väntar så här länge mellan kommandona (ändras i GUI:t)
        self.delay_s = delay_s

    def __len__(self) -> int:
        return len(self.items)

    def fill(self, pieces: List[DetectedShape]) -> None:
        """Sparar bitarna med sina positioner. Bitarna ligger still, så positionerna gäller."""
        self.items = list(pieces)
        self.running = False

    def start(self) -> bool:
        if self.items:
            self.running = True
        return self.running

    def stop(self) -> None:
        self.running = False

    def clear(self) -> int:
        """Tömmer kön. Returnerar hur många bitar som inte skickades."""
        left = len(self.items)
        self.items, self.running = [], False
        return left

    def wait_left(self, now: float) -> float:
        """Sekunder kvar av pausen efter förra kommandot."""
        return max(0.0, self.next_pick_time - now)

    def jump(self, piece: DetectedShape) -> None:
        """Flyttar biten först i kön och startar kön. Den skickas så fort pausen efter förra biten har gått."""
        if piece in self.items:
            self.items.remove(piece)
        self.items.insert(0, piece)
        self.running = True

    def nearest(self, piece: DetectedShape, max_dist_px: float) -> Optional[DetectedShape]:
        """Den sparade biten närmast piece (samma bit i en nyare frame), om den ligger närmare än max_dist_px."""
        def dist(q):
            return np.hypot(q.cx_px - piece.cx_px, q.cy_px - piece.cy_px)
        best = min(self.items, key=dist, default=None)
        return best if best is not None and dist(best) < max_dist_px else None

    def due(self, now: float) -> Optional[DetectedShape]:
        """Biten som ska skickas nu, eller None (pausad, tom, eller pausen inte slut)."""
        if self.running and self.items and now >= self.next_pick_time:
            return self.items[0]
        return None

    def mark_sent(self, now: float) -> bool:
        """Första biten är skickad: ta bort den och starta pausen. Returnerar True när kön blev tom."""
        self.items.pop(0)
        self.next_pick_time = now + self.delay_s
        if not self.items:
            self.running = False
            return True
        return False


# Skicka till roboten
class PickSender:
    def __init__(self, commands: queue.Queue, calibration: MarkerCalibration, log: CandidateLog):
        self.commands = commands             # läses av robot_worker (thread 3)
        self.calibration = calibration
        self.log = log

    def send(self, piece: DetectedShape, source: str) -> Optional[str]:
        """Lägger kommandot i kön och returnerar det, eller None om biten inte har någon mm-position."""
        # Utan ArUco-markör är X/Y pixlar och RZ = 0: skicka aldrig det till roboten
        if not piece.has_mm or not self.calibration.found:
            print(f"\n[{source}] NOT SENT: no ArUco marker, {piece.colour} {piece.shape} has no mm position")
            return None
        cmd = pick_command(piece)
        print(f"\n[{source}] Selected {piece.colour} ({piece.colour_confidence*100:.0f}%) {piece.shape} "
              f"({piece.confidence*100:.0f}%) at X={piece.x_mm} mm, Y={piece.y_mm} mm, RZ={piece.rz_deg} deg")
        print(f"[{source}] Command: {cmd.strip()}")
        self.log.next_frame()                # skriv ut alla candidates igen nästa frame
        self.commands.put(cmd)
        return cmd


# Lägena
class PickActions(Protocol):
    """Det ett läge får göra med GUI:t (lägena känner inte till Tkinter)."""
    queue: PickQueue

    def send(self, piece: DetectedShape, source: str) -> bool: ...

    def jump(self, piece: DetectedShape) -> None: ...


class PickMode(ABC):
    label = ""
    picks_colour = False             # har läget färgknappar?
    colour: Optional[str] = None     # vald färg (bara i colour pick)

    @property
    def uses_queue(self) -> bool:
        """Får kön användas just nu?"""
        return False

    def visible(self, piece: DetectedShape) -> bool:
        """Ska biten synas (kontur) och gå att klicka på?"""
        return True

    @abstractmethod
    def on_click(self, piece: DetectedShape, actions: PickActions) -> None:
        ...


class FreePick(PickMode):
    label = "Free pick (click any piece)"

    def on_click(self, piece, actions):
        actions.send(piece, "USER CLICK")            # skicka direkt


class ColourPick(PickMode):
    label = "Colour pick (queue)"
    picks_colour = True

    @property
    def uses_queue(self) -> bool:
        return self.colour is not None

    def visible(self, piece):
        # Alla syns tills en färg är vald, sedan bara den färgen
        return self.colour is None or piece.colour == self.colour

    def on_click(self, piece, actions):
        if self.colour is None:
            return
        # Klickad bit först i kön (samma bit om den redan är sparad, annars läggs den till)
        queued = actions.queue.nearest(piece, max(piece.bbox[2], piece.bbox[3]) / 2)
        actions.jump(queued if queued is not None else piece)


MODES = {"free": FreePick, "colour": ColourPick}
