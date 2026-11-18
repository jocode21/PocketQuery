"""PocketQuery — SQL validator (Phase 1, fixed 2026-11-18).

Rules (locked scope, updated 2026-11-18 per Joel):
  ALLOWED: SELECT, INSERT, UPDATE, DELETE — on the 5 fixed tables only.
  BLOCKED: DROP, TRUNCATE, ALTER, CREATE, PRAGMA, ATTACH, DETACH,
           VACUUM, REPLACE, and any multi-statement input.
  Every table / column named must exist in the real schema.
  UPDATE and DELETE MUST have a WHERE clause — a write without one
  is blocked, so a wrong prompt can never wipe a whole table.
  INSERT / UPDATE values for marks and attendance percent must be 0-100.
  Writes (INSERT / UPDATE / DELETE) need user confirmation in the
  web UI before they run — this module only classifies them; it never
  executes anything.
"""
import re

ALLOWED_OPS = ("SELECT", "INSERT", "UPDATE", "DELETE")
BLOCKED_KEYWORDS = (
    "DROP", "TRUNCATE", "ALTER", "CREATE", "PRAGMA",
    "ATTACH", "DETACH", "VACUUM", "REPLACE", "GRANT", "REVOKE",
)

SCHEMA = {
    "students": {"id", "name", "branch", "year", "email"},
    "courses": {"id", "name", "credits", "faculty_id"},
    "marks": {"student_id", "course_id", "marks"},
    "faculty": {"id", "name", "department"},
    "attendance": {"student_id", "course_id", "percent"},
}

ALL_COLUMNS = set().union(*SCHEMA.values())
ALL_TABLES = set(SCHEMA)

# SQL words that are never column names.
KEYWORDS = {
    "select", "from", "where", "and", "or", "not", "in", "like",
    "between", "order", "by", "asc", "desc", "limit", "group",
    "having", "join", "inner", "left", "outer", "on", "as",
    "distinct", "count", "avg", "sum", "min", "max", "values",
    "into", "set", "insert", "update", "delete", "is", "null",
    "case", "when", "then", "else", "end", "all",
}


def _strip_strings(sql):
    """Replace quoted string literals with '' so values can't smuggle
    keywords, table names, or semicolons past the checks."""
    return re.sub(r"'[^']*'", "''", sql)


def _first_word(sql):
    return sql.strip().split(None, 1)[0].upper().rstrip(";") if sql.strip() else ""


def _referenced_tables(body):
    tables = set()
    for m in re.finditer(
        r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([a-z_][a-z0-9_]*)",
        body, re.IGNORECASE,
    ):
        tables.add(m.group(1).lower())
    return tables


def _check_range(body, kind):
    """marks / percent values that would be *stored* must be 0-100.
    Only checks SET clauses and INSERT VALUES, never WHERE filters
    (a filter like marks > 200 is harmless — it just matches nothing)."""
    for m in re.finditer(
        r"\bSET\s+(?:[^;]*?)\b(marks|percent)\s*=\s*(-?\d+)",
        body, re.IGNORECASE,
    ):
        if not 0 <= int(m.group(2)) <= 100:
            return f"{m.group(1)} must be between 0 and 100, got {m.group(2)}."
    m = re.search(
        r"\bINSERT\s+INTO\s+(marks|attendance)\s*"
        r"(?:\(\s*([a-z_,\s]+)\)\s*)?VALUES\s*\(\s*([^)]*)\)",
        body, re.IGNORECASE,
    )
    if m:
        table, collist, values = m.group(1).lower(), m.group(2), m.group(3)
        vals = [v.strip() for v in values.split(",")]
        col = "marks" if table == "marks" else "percent"
        idx = None
        if collist:
            cols = [c.strip().lower() for c in collist.split(",")]
            if col in cols:
                idx = cols.index(col)
        else:
            idx = 2  # (student_id, course_id, marks|percent)
        if idx is not None and idx < len(vals) and vals[idx].lstrip("-").isdigit():
            if not 0 <= int(vals[idx]) <= 100:
                return f"{col} must be between 0 and 100, got {vals[idx]}."
    return None


def validate(sql, schema=SCHEMA):
    """Return (ok, kind, error). kind is SELECT/INSERT/UPDATE/DELETE,
    or None when rejected. error is a plain-English reason, or for a
    valid write, the confirmation note; None for a valid SELECT."""
    if not sql or not sql.strip():
        return False, None, "Empty query. Type a question about the college database."
    s = sql.strip()
    if "--" in s or "/*" in s:
        return False, None, "Comments are not allowed in a query."
    # One statement only: strip trailing semicolons, then any ';' left
    # means a second statement is hiding in there.
    core = s.rstrip().rstrip(";").strip()
    if ";" in core:
        return False, None, "Only one statement at a time."
    first = _first_word(s)
    if first not in ALLOWED_OPS:
        if first in BLOCKED_KEYWORDS:
            return False, None, (
                f"{first} is blocked. PocketQuery never drops or "
                "restructures tables."
            )
        return False, None, (
            "Only SELECT, INSERT, UPDATE and DELETE are allowed; "
            f"got {first}."
        )
    body = _strip_strings(s)
    # Blocked keywords anywhere in the statement, not just first word.
    for kw in BLOCKED_KEYWORDS:
        if re.search(r"\b" + kw + r"\b", body, re.IGNORECASE):
            return False, None, (
                f"{kw} is blocked. PocketQuery never drops or "
                "restructures tables."
            )
    tables = _referenced_tables(body)
    for t in tables:
        if t not in schema:
            return False, None, (
                f"I do not know the table '{t}'. I can answer about "
                "students, courses, marks, faculty, attendance."
            )
    # Qualified columns table.column must exist.
    for m in re.finditer(
        r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)", body
    ):
        t, c = m.group(1).lower(), m.group(2).lower()
        if t in schema and c not in schema[t]:
            return False, None, (
                f"Column '{c}' does not exist in table '{t}'."
            )
        if t not in schema and t not in ("students", "courses", "marks",
                                         "faculty", "attendance"):
            return False, None, (
                f"I do not know the table '{t}'. I can answer about "
                "students, courses, marks, faculty, attendance."
            )
    # Unqualified identifiers: every bare word must be a keyword, a
    # table name, or a column that belongs to a referenced table.
    # This catches e.g. SELECT foo FROM students.
    allowed_cols = set()
    for t in tables:
        allowed_cols |= schema.get(t, set())
    if not tables:
        allowed_cols = set().union(*schema.values())
    for m in re.finditer(r"\b([a-z_][a-z0-9_]*)\b", body):
        w = m.group(1).lower()
        if w in KEYWORDS or w in schema or w in ALL_TABLES:
            continue
        if w in ALL_COLUMNS:
            if w not in allowed_cols:
                tbl = sorted(tables)[0] if tables else "students"
                return False, None, (
                    f"Column '{w}' does not exist in table '{tbl}'."
                )
            continue
        return False, None, (
            f"Column '{w}' does not exist. I can answer about "
            "students, courses, marks, faculty, attendance."
        )
    # INSERT column lists must name real columns of the target table.
    m = re.search(
        r"\bINSERT\s+INTO\s+([a-z_][a-z0-9_]*)\s*\(\s*([a-z_,\s]+)\)",
        body, re.IGNORECASE,
    )
    if m:
        t = m.group(1).lower()
        for c in [c.strip().lower() for c in m.group(2).split(",")]:
            if c not in schema.get(t, set()):
                return False, None, (
                    f"Column '{c}' does not exist in table '{t}'."
                )
    # UPDATE ... SET <col> = ... must name a real column.
    if first == "UPDATE":
        m = re.search(r"\bUPDATE\s+([a-z_][a-z0-9_]*)\s+SET\s+([a-z_][a-z0-9_]*)",
                      body, re.IGNORECASE)
        if m and m.group(2).lower() not in schema.get(m.group(1).lower(), set()):
            return False, None, (
                f"Column '{m.group(2)}' does not exist in table "
                f"'{m.group(1)}'."
            )
    # UPDATE / DELETE without WHERE would wipe a whole table: block it.
    if first in ("UPDATE", "DELETE") and not re.search(
        r"\bWHERE\b", body, re.IGNORECASE
    ):
        return False, None, (
            f"{first} needs a condition (a WHERE clause). I will not "
            f"{first.lower()} a whole table without one — tell me "
            "which student, course or record you mean."
        )
    err = _check_range(body, first)
    if err:
        return False, None, err
    if first in ("INSERT", "UPDATE", "DELETE"):
        return True, first, (
            "Write query — needs your confirmation before it runs."
        )
    return True, first, None


def preview_count_sql(sql):
    """Build a SELECT COUNT(*) that shows how many rows an UPDATE or
    DELETE would affect, using its own WHERE clause. Returns None if
    the WHERE clause cannot be extracted safely."""
    body = _strip_strings(sql.strip())
    m = re.search(
        r"\bUPDATE\s+([a-z_][a-z0-9_]*)\s+SET\s+.+?\bWHERE\b\s+(.+)",
        body, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return f"SELECT COUNT(*) FROM {m.group(1)} WHERE {m.group(2).rstrip(';')}"
    m = re.search(
        r"\bDELETE\s+FROM\s+([a-z_][a-z0-9_]*)\s+WHERE\b\s+(.+)",
        body, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return f"SELECT COUNT(*) FROM {m.group(1)} WHERE {m.group(2).rstrip(';')}"
    return None


def run_readonly(con, sql):
    """Run a SELECT and return (columns, rows). Raises on bad SQL."""
    cur = con.execute(sql)
    return [d[0] for d in cur.description], cur.fetchall()


if __name__ == "__main__":
    tests = [
        ("SELECT name FROM students WHERE branch = 'CSE';", True),
        ("INSERT INTO students (id, name, branch, year, email) VALUES (51, 'Test User', 'CSE', 1, 't@x.in');", True),
        ("UPDATE marks SET marks = 90 WHERE student_id = 1 AND course_id = 2;", True),
        ("DELETE FROM students WHERE id = 51;", True),
        ("DROP TABLE students;", False),
        ("DELETE FROM students;", False),   # no WHERE -> blocked
        ("UPDATE marks SET marks = 90;", False),  # no WHERE -> blocked
        ("SELECT name FROM hostel;", False),
        ("SELECT foo FROM students;", False),
        ("TRUNCATE TABLE marks;", False),
        ("SELECT name FROM students; DROP TABLE marks;", False),
        ("UPDATE marks SET marks = 150 WHERE student_id = 1;", False),
        ("INSERT INTO marks (student_id, course_id, marks) VALUES (1, 2, 120);", False),
        ("SELECT branch, COUNT(*) FROM students GROUP BY branch;", True),
        ("SELECT students.name, marks.marks FROM students JOIN marks ON students.id = marks.student_id WHERE marks.marks > 80 ORDER BY marks.marks DESC;", True),
    ]
    bad = 0
    for sql, want in tests:
        ok, kind, err = validate(sql)
        good = ok == want
        bad += 0 if good else 1
        print(("PASS" if good else "FAIL"), kind, "|", sql, "|", err or "")
    print("failures:", bad)
    raise SystemExit(1 if bad else 0)
