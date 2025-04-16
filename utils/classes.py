

# Class descriptions for bounding boxes
class_descriptions_bb = [
    {"id": 26, "name": "speed bump", "color": [0, 97, 97]},
    {"id": 27, "name": "car", "color": [255, 0, 0]},
    {"id": 28, "name": "bus", "color": [255, 255, 0]},
    {"id": 29, "name": "tram", "color": [255, 170, 170]},
    {"id": 30, "name": "transporter", "color": [255, 170, 0]},
    {"id": 31, "name": "truck", "color": [255, 85, 170]},
    {"id": 32, "name": "cyclist with bicycle", "color": [0, 0, 255]},
    {"id": 33, "name": "bicycle without cyclist", "color": [0, 85, 255]},
    {"id": 34, "name": "motorcyclist with motorcycle", "color": [127, 0, 170]},
    {"id": 35, "name": "motorcycle without motorcyclist", "color": [255, 0, 255]},
    {"id": 36, "name": "ambulance vehicle", "color": [0, 85, 170]},
    {"id": 37, "name": "police vehicle", "color": [127, 170, 255]},
    {"id": 38, "name": "fire truck", "color": [85, 114, 172]},
    {"id": 39, "name": "traffic light", "color": [0, 170, 255]},
    {"id": 40, "name": "traffic light cyclist/pedestrian", "color": [0, 180, 255]},
    {"id": 42, "name": "adult without baby buggy or mobility support", "color": [0, 255, 0]},
    {"id": 43, "name": "child without baby buggy or mobility support", "color": [0, 212, 0]},
    {"id": 44, "name": "person with baby buggy", "color": [0, 170, 0]},
    {"id": 45, "name": "person with mobility support", "color": [0, 103, 0]},
    {"id": 46, "name": "dog", "color": [255, 170, 255]},
    {"id": 47, "name": "signalling device railrad crossing", "color": [127, 0, 85]},
    {"id": 48, "name": "trafficsign_Gefahrenstelle", "color": [0, 255, 255]},
    {"id": 49, "name": "trafficsign_Arbeitsstelle", "color": [0, 255, 255]},
    {"id": 50, "name": "trafficsign_sonstiges Gefahrenzeichen", "color": [0, 255, 255]},
    {"id": 51, "name": "trafficsign_Vorfahrt gewaehren", "color": [0, 255, 255]},
    {"id": 52, "name": "trafficsign_Halt. Vorfahrt gewaehren", "color": [0, 255, 255]},
    {"id": 53, "name": "trafficsign_Haltestelle Linienverkehr und Schulbusse", "color": [0, 255, 255]},
    {"id": 54, "name": "trafficsign_Radweg", "color": [0, 255, 255]},
    {"id": 55, "name": "trafficsign_Gehweg", "color": [0, 255, 255]},
    {"id": 56, "name": "trafficsign_Gemeinsamer Geh- und Radweg", "color": [0, 255, 255]},
    {"id": 57, "name": "trafficsign_Getrennter Rad- und Gehweg, Radweg links", "color": [0, 255, 255]},
    {"id": 58, "name": "trafficsign_Getrennter Rad- und Gehweg, Radweg rechts", "color": [0, 255, 255]},
    {"id": 59, "name": "trafficsign_Beginn einer Fussgaengerzone", "color": [0, 255, 255]},
    {"id": 60, "name": "trafficsign_Ende einer Fussgaengerzone", "color": [0, 255, 255]},
    {"id": 61, "name": "trafficsign_Begin einer Radstrasse", "color": [0, 255, 255]},
    {"id": 62, "name": "trafficsign_Ende einer Radstrasse", "color": [0, 255, 255]},
    {"id": 63, "name": "trafficsign_Verbot fuer Radverkehr", "color": [0, 255, 255]},
    {"id": 64, "name": "trafficsign_Verbot fuer Fussgaenger", "color": [0, 255, 255]},
    {"id": 65, "name": "trafficsign_Verbot der Einfahrt", "color": [0, 255, 255]},
    {"id": 66, "name": "trafficsign_Vorfahrt an naechster Kreuzung", "color": [0, 255, 255]},
    {"id": 67, "name": "trafficsign_Vorfahrtsstrasse", "color": [0, 255, 255]},
    {"id": 68, "name": "trafficsign_Ende der Vorfahrtsstrasse", "color": [0, 255, 255]},
    {"id": 69, "name": "trafficsign_Vorrang vor dem Gegenverkehr", "color": [0, 255, 255]},
    {"id": 70, "name": "trafficsign_Parken", "color": [0, 255, 255]},
    {"id": 71, "name": "trafficsign_Anfang Parken", "color": [0, 255, 255]},
    {"id": 72, "name": "trafficsign_Ende Parken", "color": [0, 255, 255]},
    {"id": 73, "name": "trafficsign_Parken auf Gehwegen, ganz in Fahrrichtung rechts", "color": [0, 255, 255]},
    {"id": 74, "name": "trafficsign_Parken auf Gehweg, ganz in Fahrrichtung rechts Anfang", "color": [0, 255, 255]},
    {"id": 75, "name": "trafficsign_Parken auf Gehweg, ganz in Fahrrichtung rechts Ende", "color": [0, 255, 255]},
    {"id": 76, "name": "trafficsign_Fahrrad frei", "color": [0, 255, 255]},
    {"id": 77, "name": "trafficsign_Beginn verkehrsberuhigter Bereich", "color": [0, 255, 255]},
    {"id": 78, "name": "trafficsign_Ende verkehrsberuhigter Bereich", "color": [0, 255, 255]},
    {"id": 79, "name": "trafficsign_Fussgaengerueberweg", "color": [0, 255, 255]},
    {"id": 80, "name": "trafficsign_Gruenpfeilschild rechts", "color": [0, 255, 255]},
    {"id": 81, "name": "trafficsign_Personenkraftwagen", "color": [0, 255, 255]},
    {"id": 82, "name": "trafficsign_Lastenfahrrad", "color": [0, 255, 255]},
    {"id": 83, "name": "trafficsign_miscellaneous traffic signs", "color": [60, 213, 255]},
    {"id": 84, "name": "trafficmarking_Fussgangerueberweg", "color": [0, 170, 255]},
    {"id": 85, "name": "trafficmarking_Radweg Markierung", "color": [0, 170, 255]},
    {"id": 86, "name": "trafficmarking_Gehweg Markierung", "color": [0, 170, 255]},
    {"id": 87, "name": "trafficmarking_Gemeinsamer Geh- und Radweg Markierung", "color": [0, 170, 255]},
    {"id": 88, "name": "trafficmarking_Getrennter Rad- und Gehweg, Radweg links Markierung", "color": [0, 170, 255]},
    {"id": 89, "name": "trafficmarking_Getrennter Rad- und Gehweg, Radweg rechts Markierung", "color": [0, 170, 255]},
    {"id": 90, "name": "trafficmarking_Haltlinie", "color": [0, 170, 255]},
]

# Class descriptions for semantic segmentation
class_descriptions_semantic = {
    0: (125, 125, 125),     # ego vehicle
    1: (255, 103, 15),      # bicycle lane
    2: (255, 170, 0),       # pedestrian path
    3: (0, 170, 127),       # shared bicycle-/pedestrian path
    4: (170, 170, 0),       # car lane
    5: (156, 170, 168),     # sealed free space
    6: (112, 112, 112),     # unsealed path
    7: (85, 255, 127),      # horizontal vegetation
    8: (255, 170, 255),     # parking space
    9: (255, 85, 127),      # railway track
    10: (152, 101, 0),      # curb stone
    11: (170, 0, 255),      # centre line
    12: (239, 239, 0),      # lane boundary
    13: (170, 0, 0),        # vertical barrier
    14: (0, 0, 255),        # vehicle
    15: (255, 0, 0),        # cyclist with bicycle
    16: (191, 0, 0),        # bicycle without cyclist
    17: (75, 207, 255),     # motorcyclist with motorcycle
    18: (56, 168, 168),     # motorcycle without motorcyclist
    19: (255, 255, 0),      # pedestrian
    20: (170, 170, 255),    # traffic sign
    21: (85, 85, 0),        # traffic light
    22: (0, 170, 0),        # vertical vegetation
    23: (170, 255, 255),    # sky
    24: (255, 240, 250),    # miscellanous static object
    25: (226, 227, 255)     # miscellanous dynamic object
}