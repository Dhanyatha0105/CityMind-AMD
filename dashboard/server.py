#!/usr/bin/env python3
"""
CityMind Dashboard — Production Server
Serves the professional inspection platform and provides API endpoints.
"""

import json
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

PROJECT_ROOT = Path(__file__).parent.parent
DEMO_DIR = PROJECT_ROOT / "output" / "demo_rich"
DASHBOARD_DIR = Path(__file__).parent

class CityMindHandler(SimpleHTTPRequestHandler):
    """Custom handler that serves dashboard files and API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        # Serve the professional platform as the default page
        if parsed.path == "/" or parsed.path == "/index":
            self.path = "/index_final.html"
            return super().do_GET()

        # API endpoints
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed.path)
            return

        # Serve drone frames from output dir
        if parsed.path.startswith("/frames/"):
            fname = parsed.path.replace("/frames/", "")
            fpath = DEMO_DIR / "frames" / fname
            if fpath.exists():
                self.send_response(200)
                ctype = "image/jpeg" if fname.endswith(".jpg") else "image/png"
                self.send_header("Content-Type", ctype)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(fpath.read_bytes())
                return
            else:
                self._send_error(404, f"Frame not found: {fname}")
                return

        # Serve overlay images from output dir
        if parsed.path.startswith("/overlays/"):
            fname = parsed.path.replace("/overlays/", "")
            fpath = DEMO_DIR / "defect_overlays" / fname
            if fpath.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(fpath.read_bytes())
                return

        super().do_GET()

    def _handle_api(self, path):
        file_map = {
            "/api/twin": "twin.json",
            "/api/defects": "defects.json",
            "/api/agents": "agents.json",
            "/api/performance": "performance.json",
            "/api/scan-history": "scan_history.json",
            "/api/structural-elements": "structural_elements.json",
            "/api/pipeline": "pipeline.json",
            "/api/complete": "demo_data_complete.json",
        }

        filename = file_map.get(path)
        if filename:
            fpath = DEMO_DIR / filename
            if fpath.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(fpath.read_bytes())
            else:
                self._send_error(404, f"Data file not found: {filename}")
        elif path == "/api/overlays":
            overlay_dir = DEMO_DIR / "defect_overlays"
            files = sorted([f.name for f in overlay_dir.glob("*.png")]) if overlay_dir.exists() else []
            self._send_json(files)
        elif path == "/api/frames":
            frames_dir = DEMO_DIR / "frames"
            files = sorted([f.name for f in frames_dir.glob("*.jpg")]) if frames_dir.exists() else []
            self._send_json(files)
        else:
            self._send_error(404, "Unknown API endpoint")

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error(self, code, msg):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": msg}).encode())

    def log_message(self, format, *args):
        # Only log errors
        if args and '404' in str(args[0]):
            sys.stderr.write(f"  [404] {args[0]}\n")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    server = HTTPServer(("0.0.0.0", port), CityMindHandler)
    print(f"")
    print(f"  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║                                                      ║")
    print(f"  ║   🏗️  CITYMIND — Infrastructure Intelligence Platform ║")
    print(f"  ║                                                      ║")
    print(f"  ║   Dashboard:  http://localhost:{port}                   ║")
    print(f"  ║   API:        http://localhost:{port}/api/twin           ║")
    print(f"  ║                                                      ║")
    print(f"  ║   Powered by AMD Ryzen AI                            ║")
    print(f"  ║                                                      ║")
    print(f"  ╚══════════════════════════════════════════════════════╝")
    print(f"")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down CityMind server.")
        server.shutdown()


if __name__ == "__main__":
    main()
