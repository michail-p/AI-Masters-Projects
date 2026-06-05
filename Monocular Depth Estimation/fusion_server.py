"""YOLO26 + DA3METRIC fusion server. Per frame: detect → metric depth → per-bbox distance,
then plan a motor command and drive the robot over BLE. Serves the phone web app over HTTPS.

Modes:
  (default)        FastAPI over HTTPS at https://<host>:<port>
                     /app/    web app (any phone: camera → wss → overlays)
                     /stream  WebSocket the web app streams JPEG frames to
                     /view    annotated MJPEG mirror for the projector
                   Planning + BLE run here (iOS browsers have no Web Bluetooth). --no-ble to skip.
  --gradio         Gradio UI at http://<host>:7860 (dev/visual check)
  --bench          Run latency benchmark and exit
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

# DA3 pins numpy<2 in its pyproject — sys.path the cloned source instead of pip-installing
# (matches NB2's approach; keeps numpy 2.x for the rest of the env).
_DA3_SRC = Path(__file__).parent / "Depth-Anything-3" / "src"
if _DA3_SRC.exists() and str(_DA3_SRC) not in sys.path:
    sys.path.insert(0, str(_DA3_SRC))

log = logging.getLogger("fusion")


# --- model + weights -------------------------------------------------------

def load_models(weights_dir: Path):
    if not _DA3_SRC.exists():
        sys.exit(f"missing DA3 source at {_DA3_SRC} — run: git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git")
    from ultralytics import YOLO
    from depth_anything_3.api import DepthAnything3

    for f in ("yolo26_best.pt", "calibration.json"):
        if not (weights_dir / f).exists():
            sys.exit(f"missing {weights_dir / f}")
    cal = json.loads((weights_dir / "calibration.json").read_text())
    calib = (float(cal["disparity_a"]), float(cal["disparity_b"]))

    if not torch.cuda.is_available():
        sys.exit("no CUDA GPU found — this stack is GPU-only (autocast + DA3 assume CUDA)")
    device = "cuda"  # GPU-only stack (RTX 4070); autocast + DA3 assume CUDA.
    # DA3's api.py sets cudnn.benchmark=False; for fixed-shape repeated inference (our case),
    # leaving it on picks the fastest conv algos and is ~3× faster.
    torch.backends.cudnn.benchmark = True

    yolo = YOLO(str(weights_dir / "yolo26_best.pt")).to(device)
    # Keep DA3 weights in fp32 — manual .half() breaks the depth head. We use torch.autocast
    # at call time instead, which is the supported mixed-precision path.
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(device).eval()
    log.info("models loaded on %s | calib (disparity affine) a=%.4f b=%.4f", device, *calib)
    return yolo, da3, calib


# --- fusion ----------------------------------------------------------------


def da3_depth(da3, rgb_np, calib):
    """Calibrated metric depth in meters, HxW float32.

    Calibration is an affine fit in disparity (inverse-depth) space:
        1/depth_true ≈ a * (1/raw) + b
    Equivalent in metric space to a hyperbolic correction; fits monocular
    depth's near-range bias much better than a single multiplicative scalar.
    """
    a, b = calib
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        pred = da3.inference([rgb_np])
    net = pred.depth[0]
    if isinstance(net, torch.Tensor):
        net = net.float().cpu().numpy()
    if net.shape != rgb_np.shape[:2]:
        net = cv2.resize(net, (rgb_np.shape[1], rgb_np.shape[0]), interpolation=cv2.INTER_LINEAR)
    return 1.0 / np.maximum(a / np.maximum(net, 1e-3) + b, 1e-3)


def detect_with_distance(yolo, da3, calib, rgb_np, conf=0.25, depth_bias=0.0):
    """Returns [{cls, name, conf, bbox: [x1,y1,x2,y2], distance_m}, …].
    Uses per-instance masks if the model is a -seg variant, otherwise falls
    back to the bbox-crop heuristic for plain detection models.
    """
    # ultralytics treats a numpy frame as BGR (it flips to RGB internally); DA3 wants RGB.
    # Feed YOLO BGR so colors match training — critical for color-cued classes like orange cones.
    det = yolo.predict(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR), conf=conf, verbose=False)[0]
    depth = da3_depth(da3, rgb_np, calib) - depth_bias
    h, w = depth.shape
    masks = det.masks.data.cpu().numpy() if det.masks is not None else None
    results = []
    for i, box in enumerate(det.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        if masks is not None:
            m = cv2.resize(masks[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            crop = depth[m]
        else:
            # fallback for detection-only models: inner 60% bbox crop
            dx, dy = (x2 - x1) * 0.2, (y2 - y1) * 0.2
            ix1, iy1, ix2, iy2 = int(x1 + dx), int(y1 + dy), int(x2 - dx), int(y2 - dy)
            if ix2 <= ix1 or iy2 <= iy1:
                ix1, iy1, ix2, iy2 = x1, y1, x2, y2
            crop = depth[iy1:iy2, ix1:ix2].ravel()
        # Lower bound only excludes no-signal/zero pixels; upper bound filters
        # sky/far-wall noise. 0.05 m floor keeps real close-range objects in play.
        crop = crop[(crop > 0.05) & (crop < 5.0)]
        # With masks: only object pixels → median is robust to depth noise.
        # With bbox: 10th percentile = "closest visible part", avoids background
        # pixels in the crop biasing the result toward farther distances.
        if crop.size:
            d = float(np.median(crop) if masks is not None else np.percentile(crop, 10))
        else:
            d = None
        results.append({
            "cls": int(box.cls[0]),
            "name": det.names[int(box.cls[0])],
            "conf": float(box.conf[0]),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "distance_m": d,
        })
    return results


def annotate(rgb_np, dets):
    out = rgb_np.copy()
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        dist = f"{d['distance_m']:.2f}m" if d["distance_m"] is not None else "?"
        label = f"{d['name']} {d['conf']:.2f} @ {dist}"
        cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


# --- planner ---------------------------------------------------------------
# Ported from the old on-phone Planner.swift. iOS browsers can't do Web Bluetooth,
# so planning + BLE moved here to the server, which already has the detections.

TURN_DISTANCE = 0.5   # m — turn away if any obstacle is closer than this
CRUISE_SPEED = 100    # PWM-ish, 0–255
TURN_SPEED = 100


def plan(dets, frame_width):
    """Detections → (ble_cmd, label). Drive forward unless something is closer than
    TURN_DISTANCE, then turn away from the more-crowded half of the frame; tie → right."""
    threats = [d for d in dets if (d["distance_m"] or float("inf")) < TURN_DISTANCE]
    if not threats:
        return f"F{CRUISE_SPEED}", f"FORWARD {CRUISE_SPEED}"
    mid = frame_width / 2
    left = sum(1 for d in threats if (d["bbox"][0] + d["bbox"][2]) / 2 < mid)
    right = len(threats) - left
    return (f"R{TURN_SPEED}", f"RIGHT {TURN_SPEED}") if left >= right else (f"L{TURN_SPEED}", f"LEFT {TURN_SPEED}")


# --- robot BLE -------------------------------------------------------------
# 16-bit UUIDs BAAD/F00D expanded to the Bluetooth SIG base, matching the firmware.

SERVICE_UUID = "0000baad-0000-1000-8000-00805f9b34fb"
CHAR_UUID    = "0000f00d-0000-1000-8000-00805f9b34fb"


class Robot:
    """Maintains a BLE link to the ESP32-C3 and writes ASCII motor commands (F/L/R/S0,
    written with response). Sends every frame to feed the firmware's 1 s watchdog.
    `state` (off|bt_off|scanning|ready) is surfaced to the web app."""

    def __init__(self):
        self.client = None
        self.state = "off"

    async def run(self):
        from bleak import BleakClient, BleakScanner
        while True:
            self.state = "scanning"
            try:
                dev = await BleakScanner.find_device_by_filter(
                    lambda d, ad: SERVICE_UUID in (ad.service_uuids or []), timeout=10.0)
                if dev is None:
                    continue
                async with BleakClient(dev) as client:
                    self.client = client
                    self.state = "ready"
                    log.info("robot BLE connected: %s", dev.address)
                    while client.is_connected:
                        await asyncio.sleep(0.5)
            except Exception as e:
                self.state = "bt_off" if "powered on" in str(e).lower() else "scanning"
                log.warning("robot BLE: %s", e)
            finally:
                self.client = None
            await asyncio.sleep(3)   # back off before rescanning (radio off / robot absent)

    async def send(self, cmd):
        if self.client:
            await self.client.write_gatt_char(CHAR_UUID, cmd.encode(), response=True)


# --- TLS (getUserMedia needs a secure context, so the page must be served over HTTPS) ---

def local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def ensure_cert(cert_dir: Path):
    """Self-signed cert for localhost + this machine's current LAN IP. Regenerated
    when the IP changed since last run — otherwise the browser rejects the new
    address as absent from the cert's SAN."""
    import datetime
    import ipaddress
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    crt, key = cert_dir / "cert.pem", cert_dir / "key.pem"
    ip = ipaddress.ip_address(local_ip())
    if crt.exists() and key.exists():
        san_ips = (x509.load_pem_x509_certificate(crt.read_bytes())
                   .extensions.get_extension_for_class(x509.SubjectAlternativeName)
                   .value.get_values_for_type(x509.IPAddress))
        if ip in san_ips:
            return crt, key
        log.info("LAN IP is now %s — regenerating cert", ip)

    cert_dir.mkdir(exist_ok=True)
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "carrobot.local")])
    san = x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ip),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(k.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(san, critical=False)
            .sign(k, hashes.SHA256()))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    log.info("generated self-signed cert in %s", cert_dir)
    return crt, key


# --- modes -----------------------------------------------------------------

def run_bench(yolo, da3, calib, sample_path: "Path | None"):  # quoted so it doesn't eval on Python 3.8
    if sample_path:
        bgr = cv2.imread(str(sample_path))
        if bgr is None:
            sys.exit(f"--bench-image: can't read {sample_path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        img = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)

    for _ in range(3):
        detect_with_distance(yolo, da3, calib, img)
    torch.cuda.synchronize()

    yolo_t, depth_t, pipe_t = [], [], []
    for _ in range(50):
        t0 = time.perf_counter()
        yolo.predict(img, verbose=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        da3_depth(da3, img, calib)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        detect_with_distance(yolo, da3, calib, img)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        yolo_t.append((t1 - t0) * 1000)
        depth_t.append((t2 - t1) * 1000)
        pipe_t.append((t3 - t2) * 1000)

    p = lambda a, q: float(np.percentile(a, q))
    print(f"YOLO      p50={p(yolo_t,50):6.1f} ms   p95={p(yolo_t,95):6.1f} ms")
    print(f"DA3       p50={p(depth_t,50):6.1f} ms   p95={p(depth_t,95):6.1f} ms")
    # End-to-end: detect_with_distance re-runs YOLO + DA3 + per-bbox aggregation.
    print(f"PIPELINE  p50={p(pipe_t,50):6.1f} ms   p95={p(pipe_t,95):6.1f} ms   -> {1000/p(pipe_t,50):.1f} FPS")


def run_gradio(yolo, da3, calib, host: str, port: int, depth_bias: float = 0.0):
    import gradio as gr

    def fn(image):
        if image is None:
            return None, "[]"
        dets = detect_with_distance(yolo, da3, calib, image, depth_bias=depth_bias)
        return annotate(image, dets), json.dumps(dets, indent=2)

    gr.Interface(
        fn=fn,
        inputs=[gr.Image(type="numpy", label="RGB frame (landscape)")],
        outputs=[gr.Image(label="annotated"), gr.Code(label="detections JSON", language="json")],
        title="YOLO26 + DA3 Metric fusion",
    ).launch(server_name=host, server_port=port)


def run_server(yolo, da3, calib, host: str, port: int, depth_bias: float = 0.0, use_ble: bool = True, conf: float = 0.25):
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn

    robot = Robot()

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(robot.run()) if use_ble else None
        yield
        if task:
            task.cancel()

    app = FastAPI(lifespan=lifespan)

    # The web app (camera + overlay client). Served same-origin so it can reach
    # wss://<this host>/stream with no mixed-content or cross-origin issues.
    app.mount("/app", StaticFiles(directory=Path(__file__).parent / "web", html=True), name="app")

    # Placeholder so /view shows something before the phone first connects.
    ph = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(ph, "waiting for phone...", (320, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2, cv2.LINE_AA)
    _, enc = cv2.imencode(".jpg", ph, [cv2.IMWRITE_JPEG_QUALITY, 70])
    # Last annotated frame for /view. Shared across all WS connections — fine for the
    # single-robot demo; a second phone would overwrite the first's view.
    latest_view = {"jpeg": enc.tobytes()}

    def decode_jpeg(buf):
        bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @app.get("/")
    def root():
        return {"ok": True, "ws": "/stream", "view": "/view", "app": "/app/"}

    @app.get("/view", response_class=HTMLResponse)
    def view_page():
        return "<body style='margin:0;background:#000'><img src='/view.mjpg' style='width:100vw;height:100vh;object-fit:contain'></body>"

    @app.get("/view.mjpg")
    def view_stream():
        async def gen():
            boundary = b"--frame\r\n"
            while True:
                jpg = latest_view["jpeg"]
                if jpg:
                    yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(1 / 30)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        # Single-slot queue holds the freshest DECODED frame for the inference worker.
        # The receiver renders /view on every incoming frame (smooth, at the phone's send
        # rate) using the most recent overlays, so /view no longer drops to the ~3-4 fps
        # inference rate — the video stays smooth; boxes/depth just refresh at inference rate.
        latest: asyncio.Queue = asyncio.Queue(maxsize=1)
        overlay = {"dets": [], "label": "—"}

        def render_view(rgb):
            ann = cv2.cvtColor(annotate(rgb, overlay["dets"]), cv2.COLOR_RGB2BGR)
            cv2.putText(ann, f"CMD: {overlay['label']}", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            ok, enc = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                latest_view["jpeg"] = enc.tobytes()

        async def receiver():
            try:
                while True:
                    rgb = decode_jpeg(await ws.receive_bytes())
                    render_view(rgb)                 # smooth /view at the phone's frame rate
                    if latest.full():
                        latest.get_nowait()
                    latest.put_nowait(rgb)
            except Exception:
                pass   # disconnect or socket error
            finally:
                await latest.put(None)   # always unblock the worker, even on a non-disconnect error

        recv_task = asyncio.create_task(receiver())
        last_shape = None
        fps_t0 = time.monotonic(); fps_n = 0
        try:
            while True:
                rgb = await latest.get()
                if rgb is None:
                    break
                try:
                    dets = await asyncio.get_running_loop().run_in_executor(
                        None, detect_with_distance, yolo, da3, calib, rgb,
                        conf, depth_bias,
                    )
                    cmd, label = plan(dets, rgb.shape[1])
                    overlay["dets"], overlay["label"] = dets, label   # receiver picks these up
                    await robot.send(cmd)
                    h, w = rgb.shape[:2]
                    if (w, h) != last_shape:
                        last_shape = (w, h)
                        if w < h:
                            log.warning("frame %dx%d is PORTRAIT — model expects landscape; hold the phone sideways", w, h)
                    fps_n += 1
                    now = time.monotonic()
                    if now - fps_t0 >= 2.0:
                        log.info("inference %.1f fps  (%dx%d)", fps_n / (now - fps_t0), w, h)
                        fps_t0 = now; fps_n = 0
                except Exception:
                    log.exception("frame failed")
                    continue
                try:
                    await ws.send_json({"detections": dets, "command": label, "robot": robot.state})
                except Exception:
                    break
        finally:
            recv_task.cancel()
            try:
                await robot.send("S0")   # operator gone — stop now instead of coasting to the watchdog
            except Exception:
                pass   # link may already be down; firmware watchdog stops it anyway

    # Warm up cudnn.benchmark autotuning — DA3 ViT-L's first inference is ~30s on the Orin;
    # do it at startup so the first real phone frame isn't a long stall.
    log.info("warming up models...")
    _warm = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    for _ in range(3):
        detect_with_distance(yolo, da3, calib, _warm)
    torch.cuda.synchronize()

    crt, key = ensure_cert(Path(__file__).parent / "certs")
    log.info("web app: https://%s:%d/app/", local_ip(), port)
    uvicorn.run(app, host=host, port=port, log_level="info",
                ssl_certfile=str(crt), ssl_keyfile=str(key))


# --- entry -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=os.environ.get("WEIGHTS_DIR", "./weights"), help="dir with yolo26_best.pt + calibration.json (default ./weights)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=None, help="default 8000 for FastAPI, 7860 for --gradio")
    ap.add_argument("--gradio", action="store_true", help="serve Gradio dev UI instead of FastAPI WS")
    ap.add_argument("--bench", action="store_true", help="run latency benchmark and exit")
    ap.add_argument("--bench-image", type=Path, default=None, help="sample frame for --bench (default: synthetic noise)")
    ap.add_argument("--depth-bias", type=float, default=0.0, help="meters to subtract from every depth pixel before per-bbox aggregation (knob for systematic over-estimate)")
    ap.add_argument("--conf", type=float, default=0.20, help="YOLO confidence threshold (default 0.20; lower = more detections / higher recall for obstacle avoidance)")
    ap.add_argument("--no-ble", action="store_true", help="don't drive the robot over BLE (test the web app without a robot present)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    weights = Path(args.weights).resolve()
    yolo, da3, calib = load_models(weights)

    if args.bench:
        run_bench(yolo, da3, calib, args.bench_image)
    elif args.gradio:
        run_gradio(yolo, da3, calib, args.host, args.port or 7860, args.depth_bias)
    else:
        run_server(yolo, da3, calib, args.host, args.port or 8000, args.depth_bias, use_ble=not args.no_ble, conf=args.conf)


if __name__ == "__main__":
    main()
