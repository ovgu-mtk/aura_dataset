

# Class descriptions for bounding boxes
class_descriptions_bb = [
    {"id": 26, "name": "speed bump", "color": [0, 97, 97], "pos": 0},
    {"id": 27, "name": "car", "color": [255, 0, 0], "pos": 1},
    {"id": 28, "name": "bus", "color": [255, 255, 0], "pos": 2},
    {"id": 29, "name": "tram", "color": [255, 170, 170], "pos": 3},
    {"id": 30, "name": "transporter", "color": [255, 170, 0], "pos": 4},
    {"id": 31, "name": "truck", "color": [255, 85, 170], "pos": 5},
    {"id": 32, "name": "cyclist with bicycle", "color": [0, 0, 255], "pos": 6},
    {"id": 33, "name": "bicycle without cyclist", "color": [0, 85, 255], "pos": 7},
    {"id": 34, "name": "motorcyclist with motorcycle", "color": [127, 0, 170], "pos": 8},
    {"id": 35, "name": "motorcycle without motorcyclist", "color": [255, 0, 255], "pos": 9},
    {"id": 36, "name": "ambulance vehicle", "color": [0, 85, 170], "pos": 10},
    {"id": 37, "name": "police vehicle", "color": [127, 170, 255], "pos": 11},
    {"id": 38, "name": "fire truck", "color": [85, 114, 172], "pos": 12},
    {"id": 39, "name": "traffic light", "color": [0, 170, 255], "pos": 13},
    {"id": 40, "name": "traffic light cyclist/pedestrian", "color": [0, 180, 255], "pos": 14},
    {"id": 42, "name": "adult without baby buggy or mobility support", "color": [0, 255, 0], "pos": 15},
    {"id": 43, "name": "child without baby buggy or mobility support", "color": [0, 212, 0], "pos": 16},
    {"id": 44, "name": "person with baby buggy", "color": [0, 170, 0], "pos": 17},
    {"id": 45, "name": "person with mobility support", "color": [0, 103, 0], "pos": 18},
    {"id": 46, "name": "dog", "color": [255, 170, 255], "pos": 19},
    {"id": 47, "name": "signalling device railrad crossing", "color": [127, 0, 85], "pos": 20},
    {"id": 48, "name": "trafficsign_Gefahrenstelle", "color": [0, 255, 255], "pos": 21},
    {"id": 49, "name": "trafficsign_Arbeitsstelle", "color": [0, 255, 255], "pos": 22},
    {"id": 50, "name": "trafficsign_sonstiges Gefahrenzeichen", "color": [0, 255, 255], "pos": 23},
    {"id": 51, "name": "trafficsign_Vorfahrt gewaehren", "color": [0, 255, 255], "pos": 24},
    {"id": 52, "name": "trafficsign_Halt. Vorfahrt gewaehren", "color": [0, 255, 255], "pos": 25},
    {"id": 53, "name": "trafficsign_Haltestelle Linienverkehr und Schulbusse", "color": [0, 255, 255], "pos": 26},
    {"id": 54, "name": "trafficsign_Radweg", "color": [0, 255, 255], "pos": 27},
    {"id": 55, "name": "trafficsign_Gehweg", "color": [0, 255, 255], "pos": 28},
    {"id": 56, "name": "trafficsign_Gemeinsamer Geh- und Radweg", "color": [0, 255, 255], "pos": 29},
    {"id": 57, "name": "trafficsign_Getrennter Rad- und Gehweg, Radweg links", "color": [0, 255, 255], "pos": 30},
    {"id": 58, "name": "trafficsign_Getrennter Rad- und Gehweg, Radweg rechts", "color": [0, 255, 255], "pos": 31},
    {"id": 59, "name": "trafficsign_Beginn einer Fussgaengerzone", "color": [0, 255, 255], "pos": 32},
    {"id": 60, "name": "trafficsign_Ende einer Fussgaengerzone", "color": [0, 255, 255], "pos": 33},
    {"id": 61, "name": "trafficsign_Begin einer Radstrasse", "color": [0, 255, 255], "pos": 34},
    {"id": 62, "name": "trafficsign_Ende einer Radstrasse", "color": [0, 255, 255], "pos": 35},
    {"id": 63, "name": "trafficsign_Verbot fuer Radverkehr", "color": [0, 255, 255], "pos": 36},
    {"id": 64, "name": "trafficsign_Verbot fuer Fussgaenger", "color": [0, 255, 255], "pos": 37},
    {"id": 65, "name": "trafficsign_Verbot der Einfahrt", "color": [0, 255, 255], "pos": 38},
    {"id": 66, "name": "trafficsign_Vorfahrt an naechster Kreuzung", "color": [0, 255, 255], "pos": 39},
    {"id": 67, "name": "trafficsign_Vorfahrtsstrasse", "color": [0, 255, 255], "pos": 40},
    {"id": 68, "name": "trafficsign_Ende der Vorfahrtsstrasse", "color": [0, 255, 255], "pos": 41},
    {"id": 69, "name": "trafficsign_Vorrang vor dem Gegenverkehr", "color": [0, 255, 255], "pos": 42},
    {"id": 70, "name": "trafficsign_Parken", "color": [0, 255, 255], "pos": 43},
    {"id": 71, "name": "trafficsign_Anfang Parken", "color": [0, 255, 255], "pos": 44},
    {"id": 72, "name": "trafficsign_Ende Parken", "color": [0, 255, 255], "pos": 45},
    {"id": 73, "name": "trafficsign_Parken auf Gehwegen, ganz in Fahrrichtung rechts", "color": [0, 255, 255], "pos": 46},
    {"id": 74, "name": "trafficsign_Parken auf Gehweg, ganz in Fahrrichtung rechts Anfang", "color": [0, 255, 255], "pos": 47},
    {"id": 75, "name": "trafficsign_Parken auf Gehweg, ganz in Fahrrichtung rechts Ende", "color": [0, 255, 255], "pos": 48},
    {"id": 76, "name": "trafficsign_Fahrrad frei", "color": [0, 255, 255], "pos": 49},
    {"id": 77, "name": "trafficsign_Beginn verkehrsberuhigter Bereich", "color": [0, 255, 255], "pos": 50},
    {"id": 78, "name": "trafficsign_Ende verkehrsberuhigter Bereich", "color": [0, 255, 255], "pos": 51},
    {"id": 79, "name": "trafficsign_Fussgaengerueberweg", "color": [0, 255, 255], "pos": 52},
    {"id": 80, "name": "trafficsign_Gruenpfeilschild rechts", "color": [0, 255, 255], "pos": 53},
    {"id": 81, "name": "trafficsign_Personenkraftwagen", "color": [0, 255, 255], "pos": 54},
    {"id": 82, "name": "trafficsign_Lastenfahrrad", "color": [0, 255, 255], "pos": 55},
    {"id": 83, "name": "trafficsign_miscellaneous traffic signs", "color": [60, 213, 255], "pos": 56},
    {"id": 84, "name": "trafficmarking_Fussgangerueberweg", "color": [0, 170, 255], "pos": 57},
    {"id": 85, "name": "trafficmarking_Radweg Markierung", "color": [0, 170, 255], "pos": 58},
    {"id": 86, "name": "trafficmarking_Gehweg Markierung", "color": [0, 170, 255], "pos": 59},
    {"id": 87, "name": "trafficmarking_Gemeinsamer Geh- und Radweg Markierung", "color": [0, 170, 255], "pos": 60},
    {"id": 88, "name": "trafficmarking_Getrennter Rad- und Gehweg, Radweg links Markierung", "color": [0, 170, 255], "pos": 61},
    {"id": 89, "name": "trafficmarking_Getrennter Rad- und Gehweg, Radweg rechts Markierung", "color": [0, 170, 255], "pos": 62},
    {"id": 90, "name": "trafficmarking_Haltlinie", "color": [0, 170, 255], "pos": 63},
]

# Class descriptions for semantic segmentation
class_descriptions_semantic = {
    0: (255, 103, 15),      # bicycle lane
    1: (255, 170, 0),       # pedestrian path
    2: (0, 170, 127),       # shared bicycle-/pedestrian path
    3: (170, 170, 0),       # car lane
    4: (156, 170, 168),     # sealed free space
    5: (112, 112, 112),     # unsealed path
    6: (85, 255, 127),      # horizontal vegetation
    7: (255, 170, 255),     # parking space
    8: (255, 85, 127),      # railway track
    9: (152, 101, 0),       # curb stone
    10: (170, 0, 255),      # centre line
    11: (239, 239, 0),      # lane boundary
    12: (170, 0, 0),        # vertical barrier
    13: (0, 0, 255),        # vehicle
    14: (255, 0, 0),        # cyclist with bicycle
    15: (191, 0, 0),        # bicycle without cyclist
    16: (75, 207, 255),     # motorcyclist with motorcycle
    17: (56, 168, 168),     # motorcycle without motorcyclist
    18: (255, 255, 0),      # pedestrian
    19: (170, 170, 255),    # traffic sign
    20: (85, 85, 0),        # traffic light
    21: (0, 170, 0),        # vertical vegetation
    22: (170, 255, 255),    # sky
    23: (255, 240, 250),    # miscellanous static object
    24: (226, 227, 255)     # miscellanous dynamic object
}


