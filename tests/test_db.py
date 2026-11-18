"""Database tests: auto-creation, fixed schema, seed data, run_sql."""
import os
import sqlite3
import tempfile

from db import create_db, get_db, run_sql

TABLES = {"students", "courses", "marks", "faculty", "attendance"}


def test_create_db_tables_and_counts(tmp_path):
    path = str(tmp_path / "college.db")
    counts = create_db(path)
    assert counts == {"students": 50, "courses": 8, "faculty": 6,
                      "marks": counts["marks"], "attendance": counts["attendance"]}
    assert counts["marks"] > 200 and counts["attendance"] > 200
    con = sqlite3.connect(path)
    got = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert TABLES <= got
    branches = {r[0] for r in con.execute(
        "SELECT DISTINCT branch FROM students")}
    assert branches == {"CSE", "ECE", "ME", "CE"}
    cols = {r[1] for r in con.execute("PRAGMA table_info(students)")}
    assert cols == {"id", "name", "branch", "year", "email"}
    con.close()


def test_get_db_creates_when_missing(tmp_path):
    path = str(tmp_path / "college.db")
    assert not os.path.exists(path)
    con = get_db(path)
    assert os.path.exists(path)
    assert con.execute("SELECT COUNT(*) FROM students").fetchone()[0] == 50
    con.close()


def test_run_sql_select_and_write(tmp_path):
    path = str(tmp_path / "college.db")
    create_db(path)
    con = sqlite3.connect(path)
    kind, cols, rows, n = run_sql(
        con, "SELECT name FROM students WHERE branch = 'CSE' ORDER BY name;")
    assert kind == "SELECT" and cols == ["name"] and n == len(rows) > 0
    kind, _, _, aff = run_sql(
        con, "INSERT INTO students (id, name, branch, year, email) "
             "VALUES (99, 'Test One', 'CSE', 2, 't@x.in');")
    assert kind == "INSERT" and aff == 1
    kind, _, _, aff = run_sql(
        con, "UPDATE students SET year = 3 WHERE id = 99;")
    assert kind == "UPDATE" and aff == 1
    kind, _, _, aff = run_sql(con, "DELETE FROM students WHERE id = 99;")
    assert kind == "DELETE" and aff == 1
    assert con.execute(
        "SELECT COUNT(*) FROM students WHERE id = 99").fetchone()[0] == 0
    con.close()
