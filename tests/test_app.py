"""Flask app tests, with the model stubbed out.

The app flow (validate -> preview -> confirm -> execute, and the
blocking rules) must be testable without waiting for a training run,
so the generator is replaced by a stub that returns fixed SQL for
fixed questions. Real-model inference is covered separately in
test_inference.py, which skips when no checkpoint exists.
"""
import json
import os
import sqlite3

import pytest

import app as app_module
import model as model_module
from db import create_db


def _stub_generate(model, vocab, text, device="cpu", max_len=64):
    q = text.lower()
    if "hostel" in q:
        return "SELECT * FROM hostel;"
    if "clear" in q:
        return "DROP TABLE students;"
    if "delete all" in q:
        return "DELETE FROM students;"
    if "add a new student" in q:
        return ("INSERT INTO students (id, name, branch, year, email) "
                "VALUES (77, 'Test New', 'CSE', 2, "
                "'test.new77@aisat.ac.in');")
    if "update marks" in q:
        return ("UPDATE marks SET marks = 88 WHERE student_id = 1 "
                "AND course_id = 2;")
    if "delete student" in q:
        return "DELETE FROM students WHERE id = 77;"
    return "SELECT name FROM students WHERE branch = 'CSE' ORDER BY name;"


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "college.db")
    create_db(db_path)
    app_module.app.config["DB_PATH"] = db_path
    app_module.app.config["HISTORY_PATH"] = str(tmp_path / "history.json")
    monkeypatch.setattr(model_module, "generate", _stub_generate)
    monkeypatch.setattr(app_module, "get_model",
                        lambda *a, **k: (object(), {}))
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_home_has_no_suggested_prompts(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.data.decode().lower()
    assert "pocketquery" in html
    assert "suggested" not in html
    assert "example prompt" not in html
    # No canned example question may appear on the home page.
    assert "show cse students with marks above 80" not in html
    # ...but the schema viewer must list the tables.
    assert "students" in html and "attendance" in html


def test_ask_select_returns_sql_and_rows(client):
    r = client.post("/ask", json={"question": "show CSE students"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True
    assert d["kind"] == "SELECT"
    assert d["sql"].startswith("SELECT")
    assert d["count"] > 0
    assert d["rows"]
    assert d["columns"] == ["name"]


def test_ask_blocks_drop_intent_before_model(client):
    r = client.post("/ask", json={"question": "drop the students table"})
    d = r.get_json()
    assert d["ok"] is False
    assert "blocked" in d["error"].lower()


def test_ask_blocks_drop_sql_from_model(client):
    r = client.post("/ask", json={"question": "clear the students table"})
    d = r.get_json()
    assert d["ok"] is False
    assert "blocked" in d["error"].lower()


def test_ask_blocks_unknown_table(client):
    r = client.post("/ask", json={"question": "show hostel fees"})
    d = r.get_json()
    assert d["ok"] is False
    assert "hostel" in d["error"]


def test_ask_blocks_delete_without_where(client):
    r = client.post("/ask", json={"question": "delete all students"})
    d = r.get_json()
    assert d["ok"] is False
    assert "WHERE" in d["error"]


def test_write_needs_confirmation_then_executes(client):
    r = client.post("/ask", json={
        "question": "add a new student Test New in CSE year 2"})
    d = r.get_json()
    assert d["ok"] is True
    assert d["needs_confirmation"] is True
    assert d["preview_rows"] == 1
    # Nothing has been inserted yet.
    assert client.application.config["DB_PATH"]
    con = sqlite3.connect(client.application.config["DB_PATH"])
    assert con.execute(
        "SELECT COUNT(*) FROM students WHERE id = 77").fetchone()[0] == 0
    con.close()
    r = client.post("/confirm", json={})
    d = r.get_json()
    assert d["ok"] is True
    assert d["affected"] == 1
    con = sqlite3.connect(client.application.config["DB_PATH"])
    assert con.execute(
        "SELECT COUNT(*) FROM students WHERE id = 77").fetchone()[0] == 1
    con.close()


def test_confirm_with_nothing_pending_fails(client):
    r = client.post("/confirm", json={})
    d = r.get_json()
    assert d["ok"] is False


def test_export_csv_after_select(client):
    client.post("/ask", json={"question": "show CSE students"})
    r = client.get("/export")
    assert r.status_code == 200
    assert "text/csv" in r.content_type
    assert r.data.decode().splitlines()[0] == "name"


def test_reset_recreates_database(client):
    db_path = client.application.config["DB_PATH"]
    con = sqlite3.connect(db_path)
    con.execute("DELETE FROM students WHERE id = 1")
    con.commit()
    con.close()
    r = client.post("/reset")
    assert r.status_code in (200, 302)
    con = sqlite3.connect(db_path)
    assert con.execute(
        "SELECT COUNT(*) FROM students").fetchone()[0] == 50
    con.close()


def test_history_is_recorded(client):
    client.post("/ask", json={"question": "show CSE students"})
    hist_path = client.application.config["HISTORY_PATH"]
    assert os.path.exists(hist_path)
    with open(hist_path) as f:
        hist = json.load(f)
    assert hist and hist[0]["question"] == "show CSE students"
    assert hist[0]["ok"] is True
