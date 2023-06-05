import os
import time
from collections import defaultdict

import pytest

CALL_TIMES = defaultdict(lambda: 0.0)


def pytest_sessionfinish(maxprint=50):
    out_str = """
Call times:
===========
"""
    keys = list(CALL_TIMES.keys())
    if len(keys) > 1:
        maxchar = max(*[len(key) for key in keys])
    else:
        maxchar = len(keys[0])
    for i, (key, item) in enumerate(
        sorted(CALL_TIMES.items(), key=lambda x: x[1], reverse=True)
    ):
        spaces = "  " + " " * (maxchar - len(key))
        out_str += f"\t{key}{spaces}{item: 4.4f}s\n"
        if i == maxprint - 1:
            break
    print(out_str)


@pytest.fixture(autouse=True)
def measure_duration(request: pytest.FixtureRequest):
    start_time = time.time()

    def fin():
        duration = time.time() - start_time
        name = request.node.name
        class_name = request.cls.__name__ if request.cls else None
        name = name.split("[")[0]
        if class_name is not None:
            name = "::".join([class_name, name])
        file = os.path.basename(request.path)
        name = f"{file}::{name}"
        CALL_TIMES[name] = CALL_TIMES[name] + duration

    request.addfinalizer(fin)
