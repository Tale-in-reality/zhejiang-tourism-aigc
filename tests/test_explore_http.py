"""explore 模式的 HTTP 回归：路由、请求方法、状态流转。

这三条端点是新流程的入口，路由写错（比如 selection 只挂在 POST 上）在
orchestrator 单测里看不出来——必须走真实 HTTP。
"""

import http.client
import importlib
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from backend.app.config import Settings
from backend.app.services.orchestrator import Orchestrator
from backend.app.storage import Storage


class ExploreHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.import_dir = tempfile.TemporaryDirectory()
        isolated = Settings(
            data_dir=Path(cls.import_dir.name),
            audio_model_cache=Path(cls.import_dir.name) / "models",
            default_provider="offline",
        )
        with patch("backend.app.config.settings", isolated):
            cls.app = importlib.import_module("backend.app.main")
        cls.addClassCleanup(cls.import_dir.cleanup)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(
            data_dir=Path(self.temp.name),
            audio_model_cache=Path(self.temp.name) / "models",
            default_provider="offline",
            asset_provider="offline",
        )
        self.settings.ensure_dirs()
        self.storage = Storage(self.settings.db_path)
        self.orchestrator = Orchestrator(self.settings, self.storage)
        for name, value in (
            ("settings", self.settings),
            ("storage", self.storage),
            ("orchestrator", self.orchestrator),
        ):
            context = patch.object(self.app, name, value)
            context.start()
            self.addCleanup(context.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.app.AppHandler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()
        self.addCleanup(self._stop_server)

    def _stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, data=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=120
        )
        try:
            connection.request(
                method,
                path,
                body=json.dumps(data).encode() if data is not None else None,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            body = response.read()
            return response.status, json.loads(body) if body else {}
        finally:
            connection.close()

    def test_explore_flow_over_http(self):
        status, task = self.request(
            "POST",
            "/api/previews",
            {
                "city": "杭州", "landmark": "西湖", "theme": "西湖春游",
                "duration": 30, "style": "诗意纪实", "shot_count": 4,
                "asset_source": "offline", "creation_mode": "explore",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(task["status"], "candidates_ready")
        self.assertEqual(task["creation_mode"], "explore")
        self.assertEqual(len(task["shot_candidates"]), 10)

        chosen = [
            candidate["id"] for candidate in task["shot_candidates"]
            if candidate["role"] in {"opening", "subject", "detail", "closing"}
        ][:4]
        status, selection = self.request(
            "PATCH",
            f"/api/tasks/{task['id']}/selection",
            {"selection": [{"candidate_id": cid, "order": index} for index, cid in enumerate(chosen, 1)]},
        )
        self.assertEqual(status, 200, selection)
        self.assertEqual(selection["status"], "selected")
        self.assertEqual(len(selection["selection"]), 4)

        status, detail = self.request("POST", f"/api/tasks/{task['id']}/detail", {})
        self.assertEqual(status, 200, detail)
        self.assertEqual(detail["shot_count"], 4)
        self.assertTrue(detail["gate"]["ok"], detail["gate"]["problems"])

        status, stored = self.request("GET", f"/api/tasks/{task['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(stored["status"], "draft")
        self.assertEqual(len(stored["shots"]), 4)
        for shot in stored["shots"]:
            self.assertTrue(shot["composed_prompt"])
            self.assertTrue(shot["intent_anchors"])

    def test_direct_mode_still_available(self):
        status, task = self.request(
            "POST",
            "/api/previews",
            {"city": "杭州", "theme": "直出脚本", "duration": 15, "shot_count": 2},
        )
        self.assertEqual(status, 200)
        self.assertEqual(task["status"], "draft")
        self.assertEqual(task["creation_mode"], "direct")
        self.assertEqual(len(task["shots"]), 2)

    def test_selection_rejects_unknown_candidate(self):
        status, task = self.request(
            "POST",
            "/api/previews",
            {"city": "杭州", "theme": "选择校验", "duration": 15, "creation_mode": "explore"},
        )
        self.assertEqual(status, 200)
        status, body = self.request(
            "PATCH",
            f"/api/tasks/{task['id']}/selection",
            {"selection": [{"candidate_id": "cx", "order": 1}]},
        )
        self.assertEqual(status, 400)
        self.assertIn("未知候选", body.get("message", ""))


if __name__ == "__main__":
    unittest.main()
