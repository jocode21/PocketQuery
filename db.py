"""PocketQuery — database layer (Phase 1).

Creates data/college.db on first run if missing, with the fixed 5-table
schema and fake seed data. No server, no password, no internet.
SQLite is built into Python; the DB is just a file.

UPDATED SCOPE (2026-11-18, per Joel): the app supports
SELECT + INSERT + UPDATE + DELETE. DROP TABLE (and TRUNCATE / ALTER /
CREATE / PRAGMA / ATTACH) is blocked — see validate.py. Writes run only
after the user confirms them in the web UI; reads need no confirmation.
"""
import os
import random
import sqlite3

BRANCHES = ["CSE", "ECE", "ME", "CE"]
YEARS = [1, 2, 3, 4]

FIRST_NAMES = [
    "Aarav", "Aditi", "Alan", "Alex", "Amal", "Ananya", "Anil", "Anju",
    "Antony", "Arjun", "Asha", "Ashwin", "Dev", "Divya", "Fathima",
    "Gokul", "Hari", "Isha", "Jefin", "Joel", "Kavya", "Kevin", "Kiran",
    "Lakshmi", "Meera", "Nikhil", "Niya", "Priya", "Rahul", "Riya",
    "Rohan", "Sandra", "Sneha", "Sreya", "Vishnu", "Zara",
]
LAST_NAMES = [
    "Nair", "Menon", "Pillai", "Iyer", "Thomas", "Mathew", "George",
    "Kurian", "Joseph", "Varghese", "Chacko", "Paul", "James", "John",
    "Kumar", "Raj", "Varma", "Krishnan", "Suresh", "Anand",
]

COURSES = [
    (1, "Data Structures", 4, 1),
    (2, "Database Management Systems", 4, 1),
    (3, "Operating Systems", 4, 2),
    (4, "Artificial Intelligence", 3, 2),
    (5, "Computer Networks", 3, 3),
    (6, "Digital Electronics", 4, 3),
    (7, "Thermodynamics", 4, 4),
    (8, "Engineering Mathematics", 4, 5),
]

FACULTY = [
    (1, "Dr. Anitha Nair", "CSE"),
    (2, "Prof. Rajesh Kumar", "CSE"),
    (3, "Dr. Meera Pillai", "ECE"),
    (4, "Prof. Suresh Varma", "ME"),
    (5, "Dr. Lakshmi Iyer", "CE"),
    (6, "Prof. George Mathew", "ECE"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    branch TEXT NOT NULL,
    year INTEGER NOT NULL,
    email TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    credits INTEGER NOT NULL,
    faculty_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS marks (
    student_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    marks INTEGER NOT NULL,
    PRIMARY KEY (student_id, course_id)
);
CREATE TABLE IF NOT EXISTS faculty (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attendance (
    student_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    percent INTEGER NOT NULL,
    PRIMARY KEY (student_id, course_id)
);
"""


def seed_students(rng):
    """50 fake students, deterministic for a given seed."""
    names = set()
    students = []
    while len(students) < 50:
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if name in names:
            continue
        names.add(name)
        sid = len(students) + 1
        branch = rng.choice(BRANCHES)
        year = rng.choice(YEARS)
        email = (
            name.lower().replace(" ", ".") + "@aisat.ac.in"
        )
        students.append((sid, name, branch, year, email))
    return students


def create_db(path, seed=7):
    """Create (or recreate) the DB at path and seed it. Returns row counts."""
    rng = random.Random(seed)
    if os.path.exists(path):
        os.remove(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.executescript(SCHEMA)
        students = seed_students(rng)
        con.executemany(
            "INSERT INTO students VALUES (?, ?, ?, ?, ?)", students
        )
        con.executemany(
            "INSERT INTO faculty VALUES (?, ?, ?)", FACULTY
        )
        con.executemany(
            "INSERT INTO courses VALUES (?, ?, ?, ?)", COURSES
        )
        marks_rows, att_rows = [], []
        for (sid, _n, _b, _y, _e) in students:
            for (cid, _cn, _cr, _f) in COURSES:
                if rng.random() < 0.7:  # each student takes ~70% of courses
                    marks_rows.append(
                        (sid, cid, rng.randint(35, 98))
                    )
                    att_rows.append(
                        (sid, cid, rng.randint(55, 100))
                    )
        con.executemany(
            "INSERT INTO marks VALUES (?, ?, ?)", marks_rows
        )
        con.executemany(
            "INSERT INTO attendance VALUES (?, ?, ?)", att_rows
        )
        con.commit()
        counts = {
            t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("students", "courses", "marks", "faculty", "attendance")
        }
        return counts
    finally:
        con.close()


def get_db(path):
    """Open an existing DB, creating + seeding it first if missing."""
    if not os.path.exists(path):
        create_db(path)
    return sqlite3.connect(path)


def run_sql(con, sql):
    """Execute one already-validated statement.

    Returns (kind, columns, rows, affected):
      SELECT -> (SELECT, [col...], [row...], len(rows))
      INSERT/UPDATE/DELETE -> (OP, [], [], cursor.rowcount)
    Raises sqlite3.Error on bad SQL. The caller must have run
    validate.validate() first; this function never checks safety.
    """
    kind = sql.strip().split(None, 1)[0].upper().rstrip(";")
    cur = con.execute(sql)
    if kind == "SELECT":
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return kind, cols, rows, len(rows)
    con.commit()
    return kind, [], [], cur.rowcount


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    counts = create_db(os.path.join(here, "data", "college.db"))
    print("college.db created:", counts)
