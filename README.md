---
title: Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Plant Disease Detection

Upload a leaf photo and the CNN model tells you which of 39 plant disease classes it belongs to, with prevention steps and supplement suggestions.

## Run locally
```
pip install -r requirements.txt
python app.py
```
Open http://localhost:7860

Place `plant_disease_model_1_latest.pt` in the project root first.
