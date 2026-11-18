"""PocketQuery — web app (Flask).

Run:  python app.py          -> http://127.0.0.1:5000
No server to install, no password, no internet. The SQLite file
data/college.db is created automatically on first run if missing.

Flow:
  SELECT: type a question -> model writes SQL -> validator checks it
          -> it runs -> you see the SQL and the result table.
  INSERT / UPDATE / DELETE (2 steps): type a prompt -> model writes
          SQL -> validator checks it -> you see the SQL plus a
          preview of how many rows it would affect -> you must click
          Confirm -> only then it runs, and you see affected rows.
  Blocked (DROP / TRUNCATE / ALTER / CREATE TABLE, UPDATE or DELETE
  without WHERE, unknown table/column, marks outside 0-100): you get
  a plain-English error and nothing runs. Nothing ever crashes the
  app, and a wrong prompt can never silently wipe data.

There is deliberately NO suggested-prompts / example-prompts section
anywhere in the UI (removed 2026-11-18 per Joel). You type free
English; the schema viewer on the side shows what you can ask about.
"""
import csv
import io
import json
import os
import re
import sqlite3
import time

from flask import (Flask, abort, jsonify, redirect, render_template,
                   request, session, url_for)

from db import create_db, get_db, run_sql
from model import get_model
from validate import SCHEMA, preview_count_sql, validate

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DB_PATH = os.path.join(DATA, "college.db")
HISTORY_PATH = os.path.join(DATA, "history.json")

app = Flask(__name__)
app.secret_key = os.environ.get("POCKETQUERY_SECRET", "pocketquery-dev-only-key")
app.config["DB_PATH"] = DB_PATH
app.config["HISTORY_PATH"] = HISTORY_PATH

# Blocked *intent* in the English prompt itself. Note: "create a new
# student" is fine (that is an INSERT) — only creating/dropping
# *tables* is blocked.
BLOCKED_PROMPT = re.compile(
    r"\b(drop|truncate|alter)\b|\bcreate\s+(table|database)\b",
    re.IGNORECASE,
)


def _history_path():
    return app.config.get("HISTORY_PATH", HISTORY_PATH)


def load_history():
    try:
        with open(_history_path()) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_history(entry):
    hist = load_history()
    hist.insert(0, entry)
    hist = hist[:100]
    os.makedirs(os.path.dirname(_history_path()) or ".", exist_ok=True)
    with open(_history_path(), "w") as f:
        json.dump(hist, f, indent=1)


def _db_path():
    return app.config.get("DB_PATH", DB_PATH)


def _ro_conn():
    return sqlite3.connect(f"file:{_db_path()}?mode=ro", uri=True)


def _respond(payload, template_ctx=None):
    """JSON for API/test clients, HTML for browsers."""
    if request.is_json or "application/json" in request.headers.get(
            "Accept", ""):
        return jsonify(payload)
    ctx = {"schema": SCHEMA, "history": load_history()[:10]}
    ctx.update(template_ctx or {})
    ctx["result"] = payload
    return render_template("index.html", **ctx)


def _metrics():
    path = os.path.join(DATA, "training_metrics.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@app.route("/")
def index():
    return render_template(
        "index.html", schema=SCHEMA, history=load_history()[:10],
        result=None, metrics=_metrics(),
    )


@app.route("/schema")
def schema():
    return jsonify(SCHEMA)


@app.route("/metrics")
def metrics():
    m = _metrics()
    if m is None:
        return render_template(
            "metrics.html", metrics=None, schema=SCHEMA,
            history=load_history()[:10],
        )
    return render_template(
        "metrics.html", metrics=m, schema=SCHEMA,
        history=load_history()[:10],
    )


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or request.form.get("question") or "")
    question = question.strip()
    if not question:
        return _respond({"ok": False, "error": "Type a question first."}), 200
    if BLOCKED_PROMPT.search(question):
        return _respond({
            "ok": False,
            "error": "That operation is blocked. PocketQuery never drops or "
                     "restructures tables — you can ask, add, update or "
                     "delete rows, but not drop a table.",
        }), 200
    try:
        model, vocab = get_model()
    except FileNotFoundError as e:
        return _respond({"ok": False, "error": str(e)}), 200
    except Exception as e:  # model file corrupt etc. — never crash
        return _respond({"ok": False,
                         "error": f"Model could not be loaded: {e}"}), 200
    try:
        from model import generate
        sql = generate(model, vocab, question)
    except Exception as e:
        return _respond({"ok": False,
                         "error": f"The model failed to answer: {e}"}), 200
    ok, kind, err = validate(sql)
    if not ok:
        save_history({"question": question, "sql": sql, "kind": None,
                      "ok": False, "error": err})
        return _respond({
            "ok": False,
            "error": err,
            "sql": sql,
            "assumed": "This is the SQL the model produced for your "
                       "question. It did not run.",
        }), 200
    # Dry run: EXPLAIN checks syntax and table/column references
    # without changing anything. A weak model can produce SQL that
    # passes the validator but does not run (e.g. a missing value
    # after '='); catch that here, before offering it to the user.
    try:
        con = _ro_conn()
        try:
            con.execute("EXPLAIN " + sql)
        finally:
            con.close()
    except sqlite3.Error as e:
        save_history({"question": question, "sql": sql, "kind": kind,
                      "ok": False,
                      "error": f"The SQL the model produced does not run: {e}"})
        return _respond({
            "ok": False,
            "error": f"The SQL the model produced does not run: {e}",
            "sql": sql,
            "assumed": "This is the SQL the model produced for your "
                       "question. It did not run — try rephrasing.",
        }), 200
    if kind == "SELECT":
        t0 = time.time()
        try:
            con = _ro_conn()
            try:
                _, cols, rows, n = run_sql(con, sql)
            finally:
                con.close()
        except sqlite3.Error as e:
            save_history({"question": question, "sql": sql, "kind": kind,
                          "ok": False, "error": f"SQLite error: {e}"})
            return _respond({"ok": False, "error": f"SQLite error: {e}",
                             "sql": sql}), 200
        elapsed = round(time.time() - t0, 3)
        session["last_select_sql"] = sql
        save_history({"question": question, "sql": sql, "kind": kind,
                      "ok": True, "rows": n})
        return _respond({"ok": True, "kind": kind, "sql": sql,
                         "columns": cols, "rows": rows, "count": n,
                         "seconds": elapsed}), 200
    # ---- Write path: preview only, never execute yet. ----
    preview = None
    count_sql = preview_count_sql(sql)
    if count_sql:
        try:
            con = _ro_conn()
            try:
                preview = con.execute(count_sql).fetchone()[0]
            finally:
                con.close()
        except sqlite3.Error as e:
            return _respond({
                "ok": False,
                "error": f"Could not preview that write: {e}",
                "sql": sql,
            }), 200
    else:
        preview = 1  # INSERT: exactly one new row
    session["pending_sql"] = sql
    session["pending_question"] = question
    return _respond({
        "ok": True, "kind": kind, "sql": sql,
        "needs_confirmation": True,
        "preview_rows": preview,
        "assumed": "Check this SQL carefully before confirming — "
                   "it will change the database.",
    }), 200


@app.route("/confirm", methods=["POST"])
def confirm():
    sql = session.pop("pending_sql", None)
    question = session.pop("pending_question", "")
    if not sql:
        return _respond({"ok": False,
                         "error": "Nothing to confirm. Generate a write "
                                  "query first."}), 200
    # Re-validate the exact SQL that was generated and stored
    # server-side — never trust a SQL string sent back by the client.
    ok, kind, err = validate(sql)
    if not ok or kind == "SELECT":
        return _respond({"ok": False, "error": err or
                         "That query is no longer valid."}), 200
    try:
        con = sqlite3.connect(_db_path())
        try:
            _, _, _, affected = run_sql(con, sql)
        finally:
            con.close()
    except sqlite3.Error as e:
        return _respond({"ok": False, "error": f"SQLite error: {e}",
                         "sql": sql}), 200
    save_history({"question": question, "sql": sql, "kind": kind,
                  "ok": True, "affected": affected})
    return _respond({"ok": True, "kind": kind, "sql": sql,
                     "affected": affected,
                     "message": f"{affected} row(s) affected."}), 200


@app.route("/export")
def export():
    sql = session.get("last_select_sql")
    if not sql:
        abort(400, "Run a SELECT question first, then export its results.")
    ok, kind, _ = validate(sql)
    if not ok or kind != "SELECT":
        abort(400, "Only the last SELECT result can be exported.")
    con = _ro_conn()
    try:
        _, cols, rows, _ = run_sql(con, sql)
    finally:
        con.close()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    w.writerows(rows)
    return app.response_class(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=pocketquery_results.csv"},
    )


@app.route("/reset", methods=["POST"])
def reset():
    create_db(_db_path())
    session.pop("pending_sql", None)
    session.pop("last_select_sql", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    get_db(_db_path())  # create + seed college.db on first run
    print("PocketQuery running at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop. The database file is",
          _db_path())
    app.run(host="127.0.0.1", port=5000, debug=False)
