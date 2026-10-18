"""Actual localhost HTTP roundtrips with constructed components, no weights."""
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from english_ai_detector.serve import DetectorServer, Limits
from tests.fixtures import PASSAGE, detector

class HTTPChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = detector()
        cls.server = DetectorServer(("127.0.0.1", 0), cls.model, Limits(max_request_bytes=16000, max_characters=2000, requests_per_minute=100))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:" + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, path, body=None, headers=None):
        request = Request(self.base + path, data=body, headers=headers or {})
        try:
            response = urlopen(request, timeout=20)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.read()

    def test_real_http_parser_and_same_object_predictions(self):
        text = PASSAGE[:1900]
        status, body = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["model_id"], self.model.model_id)
        for path, payload, headers in (
            ("/detect", json.dumps({"text": text}).encode(), {"Content-Type": "application/json"}),
            ("/detect-file", text.encode(), {"Content-Type": "application/octet-stream", "X-Filename": "synthetic.txt"}),
        ):
            status, body = self.request(path, payload, headers)
            self.assertEqual(status, 200, body)
            self.assertEqual(json.loads(body)["probabilities"], self.model.predict(text)["probabilities"])
        self.assertEqual(self.request("/")[0], 200)

    def test_error_handling_and_recovery(self):
        cases = [
            ("/detect", b'{}', {"Content-Type": "application/json"}, 400),
            ("/detect", b'{}', {"Content-Type": "text/plain"}, 415),
            ("/detect", json.dumps({"text": "x" * 2001}).encode(), {"Content-Type": "application/json"}, 413),
            ("/detect", b'x' * 16001, {"Content-Type": "application/json"}, 413),
            ("/detect-file", b'not a document', {"X-Filename": "synthetic.pdf"}, 400),
            ("/detect", b'{}', {"Origin": "https://example.org", "Content-Type": "application/json"}, 403),
        ]
        for path, payload, headers, expected in cases:
            status, body = self.request(path, payload, headers)
            self.assertEqual(status, expected, body)
            message = json.loads(body)["error"]
            self.assertTrue(message.isascii())
            self.assertEqual(self.request("/health")[0], 200)
        status, body = self.request("/detect", json.dumps({"text": PASSAGE[:1900]}).encode(), {"Content-Type": "application/json"})
        self.assertEqual(status, 200, body)

if __name__ == "__main__":
    unittest.main()
