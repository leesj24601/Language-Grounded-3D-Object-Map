"""Serve the semantic map query demo with a clickable URL."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import webbrowser


def find_free_port(preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", preferred_port)) != 0:
            return preferred_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the HTML semantic map query demo.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    demo_path = repo_root / "web" / "query_demo.html"
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo HTML not found: {demo_path}")

    port = find_free_port(args.port)
    url = f"http://127.0.0.1:{port}/web/query_demo.html"
    handler = partial(SimpleHTTPRequestHandler, directory=str(repo_root))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)

    print("\nSemantic Map Query Demo", flush=True)
    print(f"URL: {url}", flush=True)
    print("Stop: Ctrl+C\n", flush=True)

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nserver stopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
