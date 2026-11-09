import pytest
from fastapi.testclient import TestClient

import app as app_module
import settings
import variants
from app import create_app
from conftest import make_engine
from options import SETTINGS
from tokens import TokenStore

ADMIN = {"X-Internal-Token": "admin"}


def hw(ram=8000):
    return {"cpu_cores": 8, "ram_total_mb": 16000, "ram_available_mb": ram, "gpu": None}


def rows(hardware=None, current=("base", "float"), downloaded=frozenset()):
    return variants.listing(hardware or hw(), current, set(downloaded))


def get(rows_, key):
    return variants.find(rows_, key)


def test_every_variant_is_listed_smallest_first_with_an_estimate():
    result = rows()
    assert [r["id"] for r in result] == ["tiny|quantized", "tiny|float", "base|quantized", "base|float"]
    assert get(result, "tiny|quantized")["estimated_mb"] == round(26 * 1.5 + 300)
    assert all("CPU" in r["title"] for r in result)  # no GPU variant is offered


def test_a_variant_that_does_not_fit_is_refused_with_the_numbers():
    tight = rows(hw(ram=500), current=("base", "float"))
    assert get(tight, "tiny|quantized")["fits"] is True
    assert get(tight, "tiny|float")["fits"] is False and "500 MB" in get(tight, "tiny|float")["reason"]


def test_unreadable_memory_claims_nothing_and_the_model_in_use_always_fits():
    assert get(rows(hw(ram=None)), "tiny|float")["fits"] is None
    result = rows(hw(ram=10), current=("tiny", "float"), downloaded={"base|float"})
    assert get(result, "tiny|float")["fits"] is True and get(result, "tiny|float")["current"] is True
    assert get(result, "base|float")["downloaded"] is True and get(result, "base|quantized")["downloaded"] is False


@pytest.fixture
def served(tmp_path, monkeypatch):
    # 500 MB free: tiny|quantized (about 339 MB) fits, tiny|float (about 456 MB) needs more than 85% of it.
    monkeypatch.setattr(app_module.hardware, "probe", lambda: hw(ram=500))
    store = TokenStore(tmp_path / "t.json")
    config = settings.Config(store.settings_path, SETTINGS)
    config.set({"update_check_hours": 0, "warm": False})
    engine = make_engine(on_disk=lambda model, compute_type: (model, compute_type) == ("base", "float"))
    return TestClient(create_app(engine, "admin", tokens=store, config=config)), config


def test_variants_endpoint_is_admin_only_and_marks_what_is_on_disk(served):
    client, _ = served
    assert client.get("/admin/variants").status_code == 401
    body = client.get("/admin/variants", headers=ADMIN).json()
    assert "500 MB" in body["summary"] and "not measured" in body["estimate"]
    assert [r["id"] for r in body["variants"] if r["current"]] == ["base|float"]
    assert [r["id"] for r in body["variants"] if r["downloaded"]] == ["base|float"]


def test_choosing_a_variant_saves_model_and_number_format(served):
    client, config = served
    reply = client.put("/admin/variant", json={"id": "tiny|quantized"}, headers=ADMIN)
    assert reply.status_code == 200 and reply.json()["restart_required"] is True
    assert (config.get("model"), config.get("compute_type")) == ("tiny", "quantized")


def test_choosing_one_that_does_not_fit_is_refused_and_saves_nothing(served):
    client, config = served
    reply = client.put("/admin/variant", json={"id": "tiny|float"}, headers=ADMIN)
    assert reply.status_code == 409 and "500 MB" in reply.json()["detail"]
    assert (config.get("model"), config.get("compute_type")) == ("base", "float")
    assert client.put("/admin/variant", json={"id": "nope"}, headers=ADMIN).status_code == 404
