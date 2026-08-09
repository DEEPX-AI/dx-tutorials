# PP-OCRv6 Tiny Camera Application

This Python application runs a two-stage OCR pipeline:

```text
640x480 camera -> PP-OCRv6 tiny DET -> text crops -> PP-OCRv6 tiny REC -> OpenCV preview
```

It does not load or run a text-line orientation classifier. Five fixed-width recognition models are selected automatically from each detected crop's aspect ratio. Crops wider than the largest supported ratio use the ratio-25 model.

The bundled `assets/ppocrv6_tiny_dict.txt` is the official tiny recognition dictionary from PaddleOCR and matches the model's 6906 output classes after the CTC blank and space tokens are added.

## Expected models

By default, the application reads compiled models from `../outputs/paddleocr_v6`:

```text
paddleocr_v6/
├── det_fixed/det_fixed.dxnn
├── rec_fixed_ratio_2_5/rec_fixed_ratio_2_5.dxnn
├── rec_fixed_ratio_5/rec_fixed_ratio_5.dxnn
├── rec_fixed_ratio_10/rec_fixed_ratio_10.dxnn
├── rec_fixed_ratio_15/rec_fixed_ratio_15.dxnn
└── rec_fixed_ratio_25/rec_fixed_ratio_25.dxnn
```

## Python environment

Use the repository's `.venv`, which is also used by JupyterLab. Install the local DX-RT Python package into that environment by running the installation cell in the tutorial, or use:

```bash
uv pip install --python <dx-tutorials>/.venv/bin/python \
  <dx-all-suite>/dx-runtime/dx_rt/python_package
```

## Run

Run from a graphical JupyterLab terminal:

```bash
cd notebooks/T10-PaddleOCRv5/app
./run_camera.sh
```

Select another camera or change capture settings when necessary:

```bash
./run_camera.sh --camera /dev/video2 --width 640 --height 480 --fps 15
```

Press `q` or `Esc` in the OpenCV window to exit. Use `./run_camera.sh --help` to see threshold and path options.

## Japanese and Chinese text

The application uses Pillow with a CJK-capable font because OpenCV's built-in
`putText` renderer does not support these Unicode characters. By default, it
uses the bundled `assets/NotoSansJP-VariableFont_wght.ttf` font.

If you remove the bundled font, install the recommended system fallback on
Debian or Raspberry Pi OS:

```bash
sudo apt update
sudo apt install -y fonts-noto-cjk
```

You can also select another Unicode font explicitly:

```bash
./run_camera.sh --font /path/to/NotoSansCJK-Regular.ttc
```

Font support and model support are separate. The bundled PP-OCRv6 tiny REC
model and its 6,906-class output primarily cover Chinese characters and Latin
text; its dictionary does not contain Japanese hiragana or katakana. The CJK
font makes supported Chinese characters and shared kanji display correctly,
but it cannot add characters that the model was not trained to recognize. Full
Japanese OCR requires a matching Japanese or multilingual REC ONNX/DXNN model
and its corresponding character dictionary.

To test all six compiled models without opening a camera, run:

```bash
./run_camera.sh --check
```
