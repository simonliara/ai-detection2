import subprocess
import time
from typing import List, Optional

import pytest


class KillProcessOnExit:
    def __init__(self, process: subprocess.Popen):
        self.process: subprocess.Popen = process

    def __del__(self):
        self.process.kill()
        self.process.wait()


class Fixture:
    def __init__(self) -> None:
        self.ai_detection: Optional[KillProcessOnExit] = None

    def start(self, args: Optional[List[str]] = None) -> None:
        if args is None:
            args = []
        self.ai_detection = KillProcessOnExit(
            subprocess.Popen(
                ["python3", "-m", "ai_detection", "--domain-id", "47", "--yolo-path", "path", *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )


@pytest.fixture()
def fixture():
    return Fixture()


def test_integration_with_fixture(fixture: Fixture):
    fixture.start()
    time.sleep(0.1)
    assert fixture.ai_detection is not None
    assert fixture.ai_detection.process.stdout is not None
    assert fixture.ai_detection.process.poll() is None
