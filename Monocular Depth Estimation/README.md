# CarRobot — YOLO26 + DA3 metric-depth obstacle avoidance

A phone streams its camera to a laptop. Per frame the laptop runs YOLO26-Seg
detection and DA3METRIC monocular depth, fuses them into a per-object distance in
metres, plans a motor command, and drives an ESP32-C3 car over BLE. The phone is
just a camera + overlay client (iOS Safari has no Web Bluetooth, so planning and
BLE live on the laptop).

```
phone camera ──wss──> fusion_server.py ──> YOLO26-Seg ─┐
                                                       ├─> per-object distance ─> plan ─BLE─> robot
                                          DA3METRIC ───┘
```

## Demo

<video src="demo.mp4" controls width="100%"></video>

## Layout

- `fusion_server.py` — the server (detection + depth fusion, planner, BLE, HTTPS web app)
- `web/index.html` — phone client: camera → WebSocket, draws the returned overlays
- `notebooks/yolo26_train.ipynb` — NB1, trains YOLO26-Seg → `yolo26_best.pt`
- `notebooks/depth_anything_3_metric.ipynb` — NB2, fits the depth calibration → `calibration.json`
- `weights/` — `yolo26_best.pt` + `calibration.json`
- `certs/` — auto-generated self-signed TLS cert + key (regenerated when LAN IP changes)
- `Depth-Anything-3/` — upstream clone, `sys.path`'d at runtime

## Setup

Needs a CUDA GPU (the stack is GPU-only; tested on an RTX 4070).

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# DA3 source + its runtime deps. We sys.path the clone instead of `pip install -e`
# so DA3's numpy<2 pin doesn't downgrade the env.
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
pip install einops safetensors huggingface_hub addict e3nn   # DA3 imports, minus the numpy pin
```

Put the trained detector at `weights/yolo26_best.pt` (from NB1). `calibration.json`
is already there from NB2. DA3METRIC weights download from HuggingFace on first run.

## Run

```powershell
python fusion_server.py                # HTTPS server (default) → https://<laptop-ip>:8000/app/
python fusion_server.py --no-ble       # same, but don't drive a robot (web app only)
python fusion_server.py --gradio        # Gradio dev UI at http://<laptop-ip>:7860
python fusion_server.py --bench         # latency benchmark, then exit
python fusion_server.py --bench --bench-image frame.jpg  # benchmark with a real image
python fusion_server.py --conf 0.15    # lower YOLO confidence threshold (default 0.20)
```

Open `/app/` on the phone (same Wi-Fi), accept the self-signed cert, hold the phone
**landscape**, press Start. `/view` mirrors the annotated stream for a projector.

Notes:

- The cert is self-signed and regenerated when the laptop's LAN IP changes.
- `--depth-bias N` subtracts N metres from every depth pixel — a knob for a
  systematic over-estimate without re-running the calibration.
- One robot / one phone at a time: the planner, BLE link, and `/view` frame are shared.
