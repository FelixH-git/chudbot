# Author Nils Wikström niwi0007
"""Inställningar för robot-programmet. Allt som beror på uppställningen (kamera, markör, tray) ändras här."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crpnet.onnx")
MODEL_INFO_PATH = os.path.join(BASE_DIR, "models", "crpnet.json")   # klasser, färger, bildstorlek: följer med modellen
TEST_IMAGE_PATH = os.path.join(BASE_DIR, "test.png")                # används om ingen kamera hittas

# ArUco-markören
MARKER_SIZE_MM = 100.0
# Vilket hörn på markören som är (0, 0). OpenCV-ordning: 0=TL, 1=TR, 2=BR, 3=BL (markörens egen frame).
# X pekar mot nästa hörn, Y mot det föregående.
ORIGIN_CORNER = 2

# Roboten och trayen
# Vinkeln varje bit ska ha i sitt fack, i robotens tray-frame (wobjAiPlaceTray, grader, moturs sett ovanifrån).
# Uppmätt från GT_angle-fotona.
SLOT_ANGLE_ROBOT_DEG = {"Circle": 0.0, "Hexagon": -5.4, "Square": -1.7, "Star": 32.2, "Triangle": 58.4}
# Markörens frame (Y "nedåt" som i bilden) är spegelvänd mot robotens frame (Z uppåt): vinklar byter tecken.
MARKER_Y_FLIPPED = True
# Extra vridning mellan markörens X-axel och robotens X-axel. 90: triangel och hexagon hamnade 30° fel med 0
# (kvadraten syns inte, 90 mod 90 = 0). Gäller bara med markörbladet lagt som vanligt ("ID 0" mot rälsen).
MARKER_TO_ROBOT_ROT_DEG = 90.0
ROBOT_HOST, ROBOT_PORT = "0.0.0.0", 5000

# Segmentering (beror på kamerans höjd, kan ändras med sliders i GUI:t)
HSV_LOWER, HSV_UPPER = (0, 50, 50), (179, 255, 255)   # färgade bitar; bord, skuggor och papper blir svarta
MIN_AREA, MAX_AREA = 43000, 120000                     # en bit, i pixlar
PAD_FRAC = 0.25                                        # padding runt cropen (samma som när datasetet togs)
TOP_LIMIT, BOTTOM_LIMIT = 0.05, 0.85                   # ignorera balkarna upptill och robotarmen nertill
ASPECT_RANGE = (0.45, 2.2)                             # riktiga bitar är ungefär 1:1

# Filter efter modellen
MIN_CONFIDENCE = 0.85
# Minsta area för en Circle: magneterna ser ut som cirklar men är mindre.
MIN_CIRCLE_AREA = 70000

# Programmet
LOG_INTERVAL_S = 10.0          # skriv ut candidates högst så här ofta
PICK_DELAY_S = 20.0            # väntan mellan två kommandon i kön (roboten säger inte till när den är klar)
DISPLAY_SIZE = (1000, 820)     # kamerabilden får högst vara så här stor i fönstret (px)
COLOUR_BUTTONS = {"red": "#dc2626", "green": "#16a34a", "blue": "#2563eb"}
