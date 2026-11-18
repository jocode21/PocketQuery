"""PocketQuery — command-line mode.

Usage:
    python main.py                 # interactive: type questions, get SQL + results
    python main.py "show CSE students"

SELECT runs immediately. INSERT / UPDATE / DELETE show the SQL and a
preview of affected rows, then ask for confirmation (y/n) before
running. DROP and friends are blocked, as in the web app.
Type exit or quit (or Ctrl+C) to leave.
"""
import sqlite3
import sys

from db import get_db, run_sql
from model import generate, get_model
from validate import preview_count_sql, validate

DB_PATH = "data/college.db"


def handle(question, model, vocab, con, interactive=True):
    """Generate + validate + run one question. Returns a printable str."""
    from app import BLOCKED_PROMPT
    if BLOCKED_PROMPT.search(question):
        return ("Blocked: PocketQuery never drops or restructures tables.")
    try:
        sql = generate(model, vocab, question)
    except Exception as e:
        return f"The model failed to answer: {e}"
    ok, kind, err = validate(sql)
    if not ok:
        return (f"Could not run that.\nSQL the model produced: {sql}\n"
                f"Reason: {err}")
    if kind == "SELECT":
        try:
            _, cols, rows, n = run_sql(con, sql)
        except sqlite3.Error as e:
            return f"SQLite error: {e}\nSQL: {sql}"
        lines = [f"SQL: {sql}", " | ".join(cols), "-" * 40]
        lines += [" | ".join(str(v) for v in r) for r in rows[:20]]
        if n > 20:
            lines.append(f"... and {n - 20} more rows")
        lines.append(f"{n} row(s)")
        return "\n".join(lines)
    # Write path: preview, then confirm.
    preview = None
    count_sql = preview_count_sql(sql)
    if count_sql:
        try:
            preview = con.execute(count_sql).fetchone()[0]
        except sqlite3.Error:
            preview = None
    else:
        preview = 1
    out = [f"SQL: {sql}",
           f"This would affect {preview} row(s)." if preview is not None
           else "This would change the database.",
           "Check the SQL carefully before confirming."]
    if interactive:
        ans = input("Run it? (y/n): ").strip().lower()
        if ans not in ("y", "yes"):
            return "\n".join(out + ["Not run."])
    try:
        _, _, _, affected = run_sql(con, sql)
    except sqlite3.Error as e:
        return f"SQLite error: {e}\nSQL: {sql}"
    return "\n".join(out + [f"Done: {affected} row(s) affected."])


def main():
    try:
        model, vocab = get_model()
    except FileNotFoundError as e:
        print(e)
        return 1
    con = get_db(DB_PATH)
    if len(sys.argv) > 1:
        print(handle(" ".join(sys.argv[1:]), model, vocab, con,
                     interactive=False))
        return 0
    print("PocketQuery (CLI). Type a question, or 'exit' to quit.")
    print("SELECT runs at once; writes ask for confirmation first.")
    try:
        while True:
            q = input("Ask: ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break
            print(handle(q, model, vocab, con))
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
