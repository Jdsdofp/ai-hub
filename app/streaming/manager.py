"""
Video & Camera Streaming Manager.
Supports: RTSP cameras, video files (MP4), YouTube URLs, webcam.
"""
import time
import threading
import uuid
from typing import Dict, Optional, Callable

import cv2
import numpy as np
from loguru import logger


class StreamSession:
    def __init__(self, session_id: str, source: str, source_type: str,
                 company_id: int, process_frame_fn: Optional[Callable] = None):
        self.session_id = session_id
        self.source = source
        self.source_type = source_type
        self.company_id = company_id
        self.process_frame_fn = process_frame_fn
        self._cap = None
        self._running = False
        self._thread = None
        self._latest_frame = None
        self._latest_result = None
        self._frame_count = 0
        self._fps = 0.0
        self._lock = threading.Lock()

    def start(self):
        if self._running:
            return
        resolved = self._resolve_source()
        self._cap = cv2.VideoCapture(resolved)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open: {resolved}")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"[Stream {self.session_id}] Started: {self.source_type}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()

    def _resolve_source(self):
        if self.source_type == "youtube":
            try:
                import yt_dlp
                ydl_opts = {
                    "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best/bestvideo+bestaudio",
                    "quiet": True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.source, download=False)
                    url = info.get("url") or info.get("manifest_url")
                    if not url and info.get("formats"):
                        fmt = sorted(info["formats"], key=lambda f: f.get("height") or 0, reverse=True)
                        url = fmt[0].get("url")
                    if not url:
                        raise RuntimeError("No playable URL found")
                    return url
            except Exception as e:
                raise RuntimeError(f"YouTube error: {e}")
        elif self.source_type == "webcam":
            return int(self.source) if self.source.isdigit() else 0
        return self.source

    def _capture_loop(self):
        t0 = time.time()
        skip = 2
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                if self.source_type == "file":
                    self._running = False
                    break
                time.sleep(0.1)
                continue
            self._frame_count += 1
            elapsed = time.time() - t0
            if elapsed > 0:
                self._fps = self._frame_count / elapsed
            if self.process_frame_fn and self._frame_count % skip == 0:
                try:
                    annotated, result = self.process_frame_fn(frame)
                    with self._lock:
                        self._latest_frame = annotated
                        self._latest_result = result
                except Exception as e:
                    logger.error(f"[Stream] Process error: {e}")
                    with self._lock:
                        self._latest_frame = frame
            else:
                with self._lock:
                    if self._latest_frame is None:
                        self._latest_frame = frame

    @property
    def latest_frame(self):
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def latest_result(self):
        with self._lock:
            return self._latest_result

    @property
    def info(self):
        return {"session_id": self.session_id, "source": self.source,
                "source_type": self.source_type, "company_id": self.company_id,
                "running": self._running, "frame_count": self._frame_count,
                "fps": round(self._fps, 1)}


class StreamManager:
    def __init__(self):
        self._sessions: Dict[str, StreamSession] = {}

    def create_session(self, source, source_type, company_id, process_fn=None):
        sid = uuid.uuid4().hex[:12]
        session = StreamSession(sid, source, source_type, company_id, process_fn)
        self._sessions[sid] = session
        return session

    def get_session(self, sid):
        return self._sessions.get(sid)

    def stop_session(self, sid):
        s = self._sessions.pop(sid, None)
        if s:
            s.stop()

    def stop_all(self):
        for sid in list(self._sessions.keys()):
            self.stop_session(sid)

    def list_sessions(self):
        return [s.info for s in self._sessions.values()]

    def get_frame_jpeg(self, sid, quality=80):
        s = self._sessions.get(sid)
        if not s:
            return None
        frame = s.latest_frame
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()


stream_manager = StreamManager()
