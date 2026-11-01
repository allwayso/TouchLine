"""启动临时回环 HTTP 服务，验证源帧读取以及路径、Host 和写入来源边界。"""

from http.server import ThreadingHTTPServer
from pathlib import Path
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from footballer_analyzer.server import ReviewApp, handler_for
from footballer_analyzer.storage import DETECTION_SCHEMA, RowWriter, read_rows, read_yaml, write_json, write_yaml
import test_reporting


class ServerTests(unittest.TestCase):
    """本地复盘 API 的读操作与输入边界测试。"""

    def test_ball_routes_restore_results_and_reject_unknown_models(self):
        """球结果刷新可读取，未知模型不可越界，写入沿用本地来源限制。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, _ = test_reporting.ReportingTests().prepare_run(temporary)
            app = ReviewApp(temporary)
            result_dir = run_dir / "balls/ball_test"
            write_json(result_dir / "manifest.json", {"result_id": "ball_test", "status": "complete", "model_id": "football_v11n", "observed_frames": 1})
            write_json(result_dir / "frames.json", [{"frame": 0, "candidates": [[1, 1, 3, 3, 0.9]], "selected": [1, 1, 3, 3, 0.9]}, {"frame": 1, "candidates": [], "selected": None}])
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/ball-models") as response:
                    self.assertIn("football_v11n_tile", [entry["id"] for entry in json.load(response)["models"]])
                with urlopen(base + "/api/ball-job") as response:
                    self.assertEqual(json.load(response)["status"], "idle")
                with urlopen(base + "/api/runs/run_test/ball?model=football_v11n") as response:
                    self.assertIsNone(json.load(response)["frames"][1]["selected"])
                with urlopen(base + "/api/runs/run_test/ball?model=coco_v8n") as response:
                    self.assertIsNone(json.load(response))
                self.assertEqual(app.data("run_test")["ball_versions"][0]["result_id"], "ball_test")
                with self.assertRaises(HTTPError):
                    urlopen(base + "/api/runs/run_test/ball?model=../../other")
                headers = {"Content-Type": "application/json"}
                with patch.object(app, "start_ball", return_value={"status": "running"}) as start:
                    with urlopen(Request(base + "/api/runs/run_test/ball", data=b'{"model_id":"football_v11n"}', headers=headers)) as response:
                        self.assertEqual(json.load(response)["status"], "running")
                    start.assert_called_once_with("run_test", {"model_id": "football_v11n"})
                with self.assertRaises(HTTPError):
                    urlopen(Request(base + "/api/runs/run_test/ball", data=b'{}', headers={**headers, "Origin": "https://example.com"}))
            finally:
                server.shutdown()
                server.server_close()

    def test_ball_job_reentry_and_failure_preserve_previous_result(self):
        """同参任务复用，异参被拒绝；后台失败时可继续读取上一份完成结果。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, _ = test_reporting.ReportingTests().prepare_run(temporary)
            app = ReviewApp(temporary)
            entered, release = threading.Event(), threading.Event()

            def fail_tracking(run, model_id, progress, device="cpu"):
                """模拟运行期间抛错，检查任务锁与失败状态。"""
                progress("ball_detecting", 2, 10)
                entered.set()
                release.wait(5)
                raise ValueError("模拟足球失败")

            with patch("footballer_analyzer.server.track_ball_run", side_effect=fail_tracking), patch("footballer_analyzer.server.model_preset"):
                job = app.start_ball("run_test", {"model_id": "football_v11n"})
                self.assertTrue(entered.wait(5))
                self.assertEqual(app.start_ball("run_test", {"model_id": "football_v11n"})["job_id"], job["job_id"])
                with self.assertRaisesRegex(ValueError, "已有足球追踪"):
                    app.start_ball("run_test", {"model_id": "coco_v8n"})
                release.set()
                for worker in threading.enumerate():
                    if worker.name.endswith("(worker)"):
                        worker.join(timeout=5)
            self.assertEqual(app.ball_job["status"], "failed")
            self.assertEqual(app.ball_job["error"], "模拟足球失败")

    def test_material_id_survives_rename_and_changes_with_content(self):
        """素材编号随内容保持稳定，重命名不改变编号而替换内容会更新。"""
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "original.mp4"
            source.write_bytes(b"original video bytes")
            app = ReviewApp(Path(temporary) / "output", temporary)
            original_id = app.clips()[0]["material_id"]
            self.assertRegex(original_id, r"^[0-9a-f]{64}$")
            renamed = source.rename(source.with_name("renamed.mp4"))
            self.assertEqual(app.clips()[0]["material_id"], original_id)
            renamed.write_bytes(b"replacement video bytes with different size")
            self.assertNotEqual(app.clips()[0]["material_id"], original_id)

    def test_clip_preview_and_analysis_http_routes(self):
        """验证素材预览、后台状态与启动参数路由，拒绝素材越界和跨源启动。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = test_reporting.ReportingTests().prepare_run(temporary)
            video = Path(manifest["config"]["video"])
            app = ReviewApp(temporary, video.parent)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/clips") as response:
                    self.assertEqual(json.load(response)[0]["clip_id"], video.name)
                with urlopen(base + f"/api/clips/{video.name}/preview") as response:
                    self.assertEqual(response.headers["Content-Type"], "image/jpeg")
                with urlopen(base + f"/api/clips/{video.name}/metadata") as response:
                    self.assertGreater(json.load(response)["duration_s"], 0)
                for path in ["/materials", "/materials.css", "/materials.js"]:
                    with urlopen(base + path) as response:
                        self.assertEqual(response.status, 200)
                with urlopen(base + "/api/analysis-job") as response:
                    self.assertEqual(json.load(response)["status"], "idle")
                headers = {"Content-Type": "application/json"}
                body = json.dumps({"clip_id": video.name, "start_frame": 2}).encode()
                with patch.object(app, "start_analysis", return_value={"status": "running"}) as start:
                    with urlopen(Request(base + "/api/analysis", data=body, headers=headers)) as response:
                        self.assertEqual(json.load(response)["status"], "running")
                    start.assert_called_once_with({"clip_id": video.name, "start_frame": 2})
                for request in [Request(base + "/api/clips/..%5Csource.avi/preview"), Request(base + "/api/analysis", data=body, headers={**headers, "Origin": "https://example.com"})]:
                    with self.assertRaises(HTTPError):
                        urlopen(request)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_material_status_and_frames_survive_move_and_rename(self):
        """按内容关联迁移、改名后的素材，保留各步骤状态并读回源帧。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = test_reporting.ReportingTests().prepare_run(temporary)
            old_video = Path(manifest["config"]["video"])
            clips_root = Path(temporary) / "moved/clips"
            clips_root.mkdir(parents=True)
            source = old_video.rename(clips_root / "renamed.avi")
            write_json(run_dir / "balls/ball_test/manifest.json", {"status": "complete"})
            write_json(run_dir / "balls/ball_failed/manifest.json", {"status": "failed"})
            write_json(run_dir / "reports/report_test/manifest.json", {"status": "complete"})
            write_json(run_dir / "reports/report_failed/manifest.json", {"status": "failed"})
            for name, overrides in [("run_running", {"status": "running"}), ("run_lines", {"kind": "white_line_review"})]:
                write_json(Path(temporary) / name / "manifest.json", {**manifest, **overrides, "run_id": name})
            app = ReviewApp(temporary, clips_root)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/clips") as response:
                    clip = json.load(response)[0]
                self.assertEqual(clip["clip_id"], source.name)
                for key in ["tracking_runs", "calibrated_runs", "ball_runs", "report_runs"]:
                    self.assertEqual(clip[key], 1, key)
                with urlopen(base + f"/api/runs/{run_dir.name}/frames/0") as response:
                    frame = cv2.imdecode(np.frombuffer(response.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(frame.shape[:2], (48, 64))
                self.assertEqual(json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")), manifest)
                source.write_bytes(b"different content at the same material path")
                for key in ["tracking_runs", "calibrated_runs", "ball_runs", "report_runs"]:
                    self.assertEqual(app.clips()[0][key], 0, key)
                with self.assertRaisesRegex(ValueError, "内容指纹"):
                    app.run_video(manifest)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_material_status_uses_hash_before_path_and_supports_legacy_runs(self):
        """同路径替换内容不得继承完成状态，无指纹的旧记录仍可按路径读取。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = test_reporting.ReportingTests().prepare_run(temporary)
            source = Path(manifest["config"]["video"])
            app = ReviewApp(temporary, source.parent)
            self.assertEqual(app.clips()[0]["tracking_runs"], 1)
            source.write_bytes(b"replacement video content")
            self.assertEqual(app.clips()[0]["tracking_runs"], 0)
            with self.assertRaisesRegex(ValueError, "内容指纹"):
                app.run_video(manifest)
            del manifest["video_sha256"]
            write_json(run_dir / "manifest.json", manifest)
            self.assertEqual(app.clips()[0]["tracking_runs"], 1)
            self.assertEqual(app.run_video(manifest), source)

    def test_detection_cache_is_optional_and_refreshes(self):
        """旧运行返回未知检测缓存，缓存补齐后返回原始框且保留置信度。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = test_reporting.ReportingTests().prepare_run(temporary)
            app = ReviewApp(temporary)
            self.assertIsNone(app.data(run_dir.name)["detections"])
            raw_row = next(read_rows(run_dir / "tracks.parquet"))
            raw_row["confidence"] = 0.15
            with RowWriter(run_dir / "detections.parquet", DETECTION_SCHEMA) as writer:
                writer.append(raw_row)
            refreshed = app.data(run_dir.name)
            self.assertEqual(refreshed["detections"][0]["confidence"], 0.15)
            self.assertEqual(len(refreshed["tracks"]), 10)

    def test_local_api_frame_and_input_boundaries(self):
        """验证运行列表与 JPEG 帧可读，越界帧、非法路径、跨源写入和错误 Host 被拒绝。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = test_reporting.ReportingTests().prepare_run(temporary)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(ReviewApp(temporary)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/workspace") as response:
                    workspace = response.read()
                with urlopen(base + "/") as response:
                    self.assertIn(b"CHANGE THE WAY", response.read())
                for path in ["/?run=run_test", "/?clip=synthetic_clip"]:
                    with urlopen(base + path) as response:
                        self.assertIn("球员追踪实验台".encode(), response.read())
                for path in ["/home.css", "/home.js", "/home-assets/Barlow-Regular.ttf", "/home-assets/tracking.webp", "/home-assets/OFL.txt"]:
                    with urlopen(base + path) as response:
                        self.assertEqual(response.status, 200)
                        self.assertTrue(response.read())
                for path in ["/home-assets/..%2Fserver.py", "/home-assets/missing.webp"]:
                    with self.assertRaises(HTTPError) as raised:
                        urlopen(base + path)
                    self.assertEqual(raised.exception.code, 404)
                with urlopen(base + "/field-review") as response:
                    field_review = response.read()
                self.assertIn("球员追踪实验台".encode(), workspace)
                self.assertNotIn(b"review-canvas", workspace)
                self.assertIn("场内轨迹校对".encode(), field_review)
                self.assertIn(b"review-canvas", field_review)
                for path, expected in [("/field-review", "场内轨迹校对"), ("/field-review.js", "showFrame"), ("/field-review.css", "review-layout")]:
                    with urlopen(base + path) as response:
                        self.assertIn(expected.encode(), response.read())
                with urlopen(base + "/api/runs") as response:
                    self.assertEqual(json.load(response)[0]["run_id"], "run_test")
                with urlopen(base + "/api/runs/run_test/frames/4") as response:
                    frame = cv2.imdecode(np.frombuffer(response.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    self.assertEqual(frame.shape[:2], (48, 64))
                for suffix in ["../manifest.json", "run_test/frames/99"]:
                    with self.assertRaises(HTTPError):
                        urlopen(base + "/api/runs/" + suffix)
                foreign_request = Request(base + "/api/runs/run_test/players", data=b'{}', headers={"Content-Type": "application/json", "Origin": "https://example.com"})
                with self.assertRaises(HTTPError) as raised:
                    urlopen(foreign_request)
                self.assertIn("跨来源", raised.exception.read().decode())
                bad_host_request = Request(base + "/api/runs", headers={"Host": "example.com"})
                with self.assertRaises(HTTPError):
                    urlopen(bad_host_request)
            finally:
                server.shutdown()
                server.server_close()

    def test_white_line_evidence_and_review_are_bound_to_calibration_version(self):
        """透明掩膜绑定标定版本，旧版本不可误确认，非法版本路径被拒绝。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, _ = test_reporting.ReportingTests().prepare_run(temporary)
            calibration = read_yaml(run_dir / "calibration.yaml")
            calibration.update(mode="white_lines", calibration_id="lines_test", reviewed=False)
            version_dir = run_dir / "calibrations/lines_test"
            write_json(version_dir / "frames.json", {"0": {"status": "insufficient_geometry", "homography": None}})
            write_yaml(version_dir / "calibration.yaml", calibration)
            write_yaml(run_dir / "calibration.yaml", calibration)
            (version_dir / "masks").mkdir()
            cv2.imwrite(str(version_dir / "masks/0.png"), np.zeros((48, 64, 4), np.uint8))
            app = ReviewApp(temporary)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}/api/runs/run_test"
            try:
                with urlopen(base + "/white-lines/0.png?version=lines_test") as response:
                    self.assertEqual(response.headers["Content-Type"], "image/png")
                with self.assertRaises(HTTPError):
                    urlopen(base + "/white-lines/0.png?version=../../manifest")
                with self.assertRaisesRegex(ValueError, "版本"):
                    app.review_calibration("run_test", {"calibration_id": "lines_old", "reviewed": True})
                app.review_calibration("run_test", {"calibration_id": "lines_test", "reviewed": True})
                self.assertTrue(app.data("run_test")["calibration"]["reviewed"])
                self.assertGreater(len(app.data("run_test")["pitch_lines"]), 10)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
