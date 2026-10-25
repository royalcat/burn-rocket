#!/usr/bin/env python3
"""Capture OTLP/HTTP trace export bodies for the OpenTelemetry smoke test.

`serve` posts protobuf span batches to `<endpoint>/v1/traces`. This development
helper writes each POST body and its headers into the output directory, so the
protobuf can be inspected without an OTLP proto checkout:

    python3 tools/otel_capture.py --port 4318 --out /tmp/otel-capture
    protoc --decode_raw < /tmp/otel-capture/*.bin

A batch normally contains the `http.request` span, its `inference` child and
the per-request log event. It answers 200 to every POST; use `--quiet` to hide
the one line printed per batch.
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4318, help="listen port")
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/otel-capture"), help="output directory"
    )
    parser.add_argument("--quiet", action="store_true", help="do not print received batches")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length)
            stamp = time.time_ns()
            (args.out / f"{stamp}.bin").write_bytes(body)
            (args.out / f"{stamp}.json").write_text(
                json.dumps({"path": self.path, "headers": dict(self.headers)}) + "\n"
            )
            if not args.quiet:
                print(f"{stamp}: {length} bytes {self.path}", flush=True)
            self.send_response(200)
            self.send_header("content-type", "application/x-protobuf")
            self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"OTLP/HTTP capture on http://127.0.0.1:{args.port} -> {args.out}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
