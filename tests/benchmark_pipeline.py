"""Read-only live RGB/AI/preview timing probe; no recorder or production writes.

Run from the project root: hk_env/bin/python tests/benchmark_pipeline.py --seconds 60
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import tempfile
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.camera_manager import CameraManager
from core.selective_camera import MainCamera
from core.selective_crypto import protect
from core.selective_recognition import Classifier, Policy


def rss_mib():
    with open('/proc/self/status', encoding='ascii') as source:
        for line in source:
            if line.startswith('VmRSS:'):
                return round(int(line.split()[1]) / 1024, 1)
    return None


def temperature():
    path = Path('/sys/class/thermal/thermal_zone0/temp')
    return round(int(path.read_text()) / 1000, 1) if path.exists() else None


def summary(values):
    ordered = sorted(values)
    return {'mean_ms': round(statistics.mean(values), 2),
            'p95_ms': round(ordered[int((len(ordered) - 1) * .95)], 2),
            'max_ms': round(ordered[-1], 2)} if values else None


def timed_call(obj, name, key, timings):
    original = getattr(obj, name)
    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timings[key].append((time.perf_counter() - started) * 1000)
    setattr(obj, name, wrapped)


def run(seconds, ort_threads=0):
    scratch_parent = Path('tests/runtime').resolve()
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='perf-', dir=scratch_parent) as scratch:
        os.environ['SRX_DATA_ROOT'] = scratch
        camera = MainCamera(config)
        camera.gallery_path = 'security_system.db'
        adapter = camera._adapter()
        if ort_threads:
            import gc
            import onnxruntime as ort
            from insightface.model_zoo.arcface_onnx import ArcFaceONNX
            model_file = adapter.recognizer.model_file
            options = ort.SessionOptions()
            options.intra_op_num_threads = ort_threads
            session = ort.InferenceSession(model_file, sess_options=options,
                                           providers=['CPUExecutionProvider'])
            recognizer = ArcFaceONNX(model_file=model_file, session=session)
            recognizer.prepare(ctx_id=-1)
            adapter.recognizer = recognizer
            camera.recognizer = recognizer
            gc.collect()
        capture = CameraManager(config=config)
        timings = defaultdict(list)
        timed_call(adapter.detector, 'detect', 'scrfd', timings)
        timed_call(adapter.recognizer, 'get_feat', 'arcface', timings)
        timed_call(adapter, 'cosine', 'cosine', timings)
        classifier = Classifier(Policy(track_iou=float(config.FACE_AUTH_TRACK_IOU)))
        states = {}; previous = []; last_detected = -1e9
        count = 0; read_failures = 0; ai_failures = 0; unique = set()
        authorized_frames = 0; detected_frames = 0
        capture.open()
        selected = dict(capture.selected)
        # Warm-up stays outside all reported latency samples.
        for _ in range(10):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('warm-up camera read failed')
            adapter.observe(frame, -1)
        timings.clear()
        started = time.monotonic()
        cpu_start = resource.getrusage(resource.RUSAGE_SELF)
        rss_start = rss_mib(); temp_start = temperature()
        try:
            while time.monotonic() - started < seconds:
                tick = time.monotonic()
                t = time.perf_counter(); ok, frame = capture.read()
                timings['read'].append((time.perf_counter() - t) * 1000)
                if not ok:
                    read_failures += 1
                    capture.reconnect()
                    continue
                width = int(config.PROCESS_WIDTH)
                if width and frame.shape[1] > width:
                    height = int(frame.shape[0] * width / frame.shape[1]) // 2 * 2
                    frame = cv2.resize(frame, (width // 2 * 2, height))
                t = time.perf_counter()
                try:
                    detected = adapter.observe(frame, count)
                except Exception:
                    ai_failures += 1
                    raise
                timings['observe'].append((time.perf_counter() - t) * 1000)
                now = time.monotonic()
                if detected:
                    last_detected = now; detected_frames += 1
                t = time.perf_counter()
                observations = camera._hold_detector_dropouts(detected, previous, last_detected, now)
                faces = classifier.classify(observations, count)
                states = camera._update_display_authorization(faces, observations, states, now)
                previous = list(observations)
                timings['track'].append((time.perf_counter() - t) * 1000)
                authorized_frames += int(any(f.get('authorized') for f in faces))
                t = time.perf_counter(); protected = protect(frame, faces)[0]
                timings['protect'].append((time.perf_counter() - t) * 1000)
                t = time.perf_counter(); ok, jpeg = cv2.imencode('.jpg', protected)
                timings['jpeg'].append((time.perf_counter() - t) * 1000)
                if not ok:
                    raise RuntimeError('JPEG encode failed')
                if count % 30 == 0:
                    unique.add(hashlib.sha256(jpeg.tobytes()).digest())
                timings['total'].append((time.monotonic() - tick) * 1000)
                count += 1
                time.sleep(max(0, 1 / float(config.SELECTIVE_FPS) - (time.monotonic() - tick)))
        finally:
            capture.release()
        elapsed = time.monotonic() - started
        cpu_end = resource.getrusage(resource.RUSAGE_SELF)
        result = {'seconds': round(elapsed, 1), 'frames': count,
                  'processing_fps': round(count / elapsed, 2),
                  'ort_threads': ort_threads or 'default',
                  'camera': selected, 'gallery_entries': len(adapter.users),
                  'read_failures': read_failures, 'ai_failures': ai_failures,
                  'unique_sample_frames': len(unique),
                  'detected_frames': detected_frames,
                  'authorized_frames': authorized_frames,
                  'rss_mib': [rss_start, rss_mib()],
                  'cpu_percent_one_core': round(100 * (
                      cpu_end.ru_utime + cpu_end.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime
                  ) / elapsed, 1),
                  'temp_c': [temp_start, temperature()],
                  'timings': {key: summary(value) for key, value in timings.items()}}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--ort-threads', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800:
        parser.error('seconds must be between 1 and 1800')
    if args.ort_threads not in (0, 1, 2, 3, 4):
        parser.error('ort-threads must be 0..4')
    run(args.seconds, args.ort_threads)
