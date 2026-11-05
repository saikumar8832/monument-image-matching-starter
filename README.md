# Monument image matching - educational starter

This is a **new sample created by Sai's AI assistant** in November 2026. It is not a recovered copy of Sai's lost monument project and is not evidence that he wrote this new code. Sai's original project was described separately as an Android application involving CNN and augmented reality; this sample does not reproduce those capabilities.

## What this demo does

This local Python command-line experiment stores user-labelled reference images and manually entered descriptions in SQLite. It finds the closest reference for a query image using a small fixed convolution, ReLU and pooling feature extractor and nearest-neighbour comparison. The filters are **hand-designed, not learned**: there is no trained CNN, pretrained model, monument dataset, Android app, augmented reality, 3D reconstruction, or historical database. Matching is a simple educational image-similarity demonstration, not reliable identification. A nearest result is returned even when none is correct, so a person must check it; distance is not a confidence score.

Optionally it can save a side-by-side PNG comparing the query with its nearest reference thumbnail. This is a 2D comparison, not augmented reality or 3D visualization.

No real monument photos, labels, descriptions or model weights are included. Use images and descriptions you are allowed to use, and verify historical facts yourself. Avoid sensitive images. The tool runs locally and has no network code; it stores image-derived features, a reference thumbnail, filename, label and description in a local SQLite database. It does not save the full-resolution original in that database.

## Setup and run

Python 3.10+ is required. In a virtual environment, install Pillow and NumPy (see requirements.txt), then run:

```sh
python -m pip install -r requirements.txt
python monuments.py --db my_catalog.sqlite3 add path/to/reference.jpg --label "Example monument" --details "Description entered by the user"
python monuments.py --db my_catalog.sqlite3 match path/to/query.jpg --output comparison.png
python -m unittest discover -s tests -v
```

Image loading is limited to 20 million pixels. The matcher resizes and crops each image, so lighting, framing and angle can change results. The demonstration tests use invented silhouettes, not photographs or an accuracy benchmark. It has not been evaluated on real monuments and must not be represented as a working trained monument-recognition or AR product.

To improve it responsibly, use an appropriately licensed, labeled dataset, split train/test images by source, train or select a properly licensed model, report held-out results and failure cases, and separately build and test any visualization or AR UI.
