import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timezone


def test_time_endpoint():
    client = TestClient(app)

    response = client.get("/time")

    assert response.status_code == 200
    current_time = datetime.fromisoformat(response.json()["timestamp"])
    expected_timezone_offset = (timezone.utc - datetime.timezone(None)).seconds // 3600

    assert current_time.year == 2023, "Year mismatch"
    assert abs(current_time.minute - 0) < 1, "Minute mismatch"
    assert abs(current_time.second - 0) < 1, "Second mismatch"
    assert (
        current_time.hour + expected_timezone_offset == 0
    ), "Incorrect timezone offset"
