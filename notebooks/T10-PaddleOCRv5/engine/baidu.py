import logging
from typing import List
from dataclasses import dataclass
from PIL import Image, ImageOps

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

det_preprocess = [           
            {
                "resize": {
                    "mode": "default",
                    "size": [480, 480],
                }
            },
            {
                "div": {
                    "x": 255
                }
            },
            {
                "normalize": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                }
            },
            {
                "transpose": {"axis": [2, 0, 1]}
            },
        ]

cls_preprocess = [
            {
                "resize": {
                    "mode": "default",
                    "size": [48, 192],
                }
            },
            {
                "div": {
                    "x": 255
                }
            },
            {
                "normalize": {
                    "std": [0.5, 0.5, 0.5],
                    "mean": [0.5, 0.5, 0.5],
                }
            },
            {
                "transpose": {"axis": [2, 0, 1]}
            },
        ]

rec_preprocess = [
            {
                "resize": {
                    "mode": "default",
                    "size": [48, 320],
                },
            },
            {
                "div": {
                    "x": 255
                }
            },
            {
                "normalize": {
                    "std": [0.5, 0.5, 0.5],
                    "mean": [0.5, 0.5, 0.5],
                }
            },
            {
                "transpose": {"axis": [2, 0, 1]}
            },
        ]

def add_white_border(img: Image):
    border_width = 200
    border_color = (255, 255, 255)
    img_with_border = ImageOps.expand(img, border=border_width, fill=border_color)
    return img_with_border

def poly2bbox(poly):
    L = poly[0]
    U = poly[1]
    R = poly[2]
    D = poly[5]
    L, R = min(L, R), max(L, R)
    U, D = min(U, D), max(U, D)
    bbox = [L, U, R, D]
    return bbox

@dataclass
class OCRResult:
    """Class to store OCR detection and recognition results"""
    text: str
    confidence: float
    bbox: List[List[float]]
    type: str = "text"

