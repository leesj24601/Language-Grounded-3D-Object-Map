from __future__ import annotations

import json
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


class GitHubPagesEntrypointTest(unittest.TestCase):
    def test_root_entrypoint_opens_demo_and_demo_data(self) -> None:
        handler = partial(QuietHandler, directory=str(REPO_ROOT))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            root_url = f"http://127.0.0.1:{server.server_port}/"
            with urlopen(root_url) as response:
                entrypoint = response.read().decode("utf-8")

            redirect_target = "./web/query_demo.html"
            self.assertIn(f"url={redirect_target}", entrypoint)

            demo_url = urljoin(root_url, redirect_target)
            with urlopen(demo_url) as response:
                demo = response.read().decode("utf-8")
            self.assertIn('id="queryForm"', demo)

            map_url = urljoin(
                demo_url,
                "../outputs/gt_aligned_10_label/maps/"
                "41098076_semantic_map_100frames_text035.json",
            )
            with urlopen(map_url) as response:
                map_data = json.load(response)
            self.assertGreater(len(map_data["objects"]), 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
