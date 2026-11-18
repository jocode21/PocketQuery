"""Validator tests: the safety rules the whole project depends on."""
from validate import validate


def test_allows_select_insert_update_delete():
    cases = [
        "SELECT name FROM students WHERE branch = 'CSE';",
        "INSERT INTO students (id, name, branch, year, email) VALUES (51, 'A', 'CSE', 1, 'a@x.in');",
        "UPDATE marks SET marks = 90 WHERE student_id = 1 AND course_id = 2;",
        "DELETE FROM students WHERE id = 51;",
        "SELECT branch, COUNT(*) FROM students GROUP BY branch;",
    ]
    for sql in cases:
        ok, kind, _ = validate(sql)
        assert ok, sql
        assert kind in ("SELECT", "INSERT", "UPDATE", "DELETE")


def test_blocks_drop_truncate_alter_create():
    for sql in [
        "DROP TABLE students;",
        "DROP DATABASE college;",
        "TRUNCATE TABLE marks;",
        "ALTER TABLE students ADD COLUMN x TEXT;",
        "CREATE TABLE foo (id INTEGER);",
        "SELECT name FROM students; DROP TABLE marks;",
        "SELECT * FROM students WHERE name = 'x'; DELETE FROM marks;",
    ]:
        ok, kind, err = validate(sql)
        assert not ok, sql
        assert kind is None
        assert err


def test_blocks_update_delete_without_where():
    for sql in [
        "DELETE FROM students;",
        "DELETE FROM marks;",
        "UPDATE marks SET marks = 90;",
        "UPDATE students SET branch = 'CSE';",
    ]:
        ok, _, err = validate(sql)
        assert not ok, sql
        assert "WHERE" in err


def test_blocks_unknown_table_and_column():
    for sql in [
        "SELECT name FROM hostel;",
        "SELECT foo FROM students;",
        "SELECT students.nope FROM students;",
        "INSERT INTO students (id, nope) VALUES (1, 2);",
        "UPDATE students SET nope = 1 WHERE id = 1;",
    ]:
        ok, _, _ = validate(sql)
        assert not ok, sql


def test_blocks_marks_out_of_range():
    for sql in [
        "UPDATE marks SET marks = 150 WHERE student_id = 1;",
        "UPDATE marks SET marks = -5 WHERE student_id = 1;",
        "INSERT INTO marks (student_id, course_id, marks) VALUES (1, 2, 120);",
        "INSERT INTO attendance (student_id, course_id, percent) VALUES (1, 2, 101);",
    ]:
        ok, _, _ = validate(sql)
        assert not ok, sql


def test_allows_marks_at_boundaries():
    for sql in [
        "UPDATE marks SET marks = 0 WHERE student_id = 1;",
        "UPDATE marks SET marks = 100 WHERE student_id = 1;",
    ]:
        ok, _, _ = validate(sql)
        assert ok, sql


def test_rejects_empty_and_multi_statement():
    assert not validate("")[0]
    assert not validate("   ")[0]
    assert not validate("SELECT * FROM students; SELECT * FROM marks;")[0]


def test_write_returns_confirmation_note():
    ok, kind, err = validate(
        "DELETE FROM students WHERE id = 51;")
    assert ok and kind == "DELETE"
    assert err and "confirmation" in err.lower()
    ok, kind, err = validate("SELECT * FROM students;")
    assert ok and kind == "SELECT"
    assert err is None
