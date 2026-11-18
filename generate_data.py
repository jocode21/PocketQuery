"""PocketQuery — dataset generator.

Generates ~16,000 English -> SQL examples from templates + slot filling,
covering the fixed college schema and the UPDATED scope (2026-11-18):
SELECT + INSERT + UPDATE + DELETE. DROP and friends are never generated.
Targets: 12,000 SELECT / 1,500 INSERT / 1,500 UPDATE / 1,000 DELETE.
Split (shuffled, seed 42): 80% train / 10% val / 10% test.

Why so many templates: a template with only one small slot (e.g. branch,
4 values x 5 phrasings = 20 unique examples) saturates fast. The
generator therefore samples templates at random until each target is
reached, and the template list leans on multi-slot combinations
(branch+marks, student+course, marks BETWEEN) whose unique space is
in the thousands. If a target cannot be reached the generator prints
a warning instead of silently writing a short file.

Usage:  python generate_data.py
Outputs: data/dataset.jsonl, data/train.jsonl, data/val.jsonl,
          data/test.jsonl, data/vocab.json
Each line: {"input": "...", "sql": "...", "op": "SELECT"|...}
"""
import json
import os
import random
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import create_db  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RNG = random.Random(42)

NEW_FIRST = ["Asha", "Bibin", "Chinchu", "Devika", "Eldho", "Femi",
             "Greeshma", "Harith", "Irene", "Jibin", "Liya", "Manu",
             "Nandana", "Roshni", "Sahil", "Thara", "Vimal", "Yadhu"]
NEW_LAST = ["Nair", "Menon", "Thomas", "Mathew", "George", "Kurian",
            "Joseph", "Paul", "James", "Kumar", "Raj", "Varma"]
NEW_COURSES = ["Machine Learning", "Web Development", "Cloud Computing",
               "Cyber Security", "Data Science", "Embedded Systems",
               "Robotics", "Software Engineering"]


def load_reference():
    tmp = os.path.join(DATA, "_ref.db")
    create_db(tmp, seed=7)
    con = sqlite3.connect(tmp)
    students = [r[0] for r in con.execute("SELECT name FROM students")]
    sids = [r[0] for r in con.execute("SELECT id FROM students")]
    courses = [r[0] for r in con.execute("SELECT name FROM courses")]
    cids = [r[0] for r in con.execute("SELECT id FROM courses")]
    faculty = [r[0] for r in con.execute("SELECT name FROM faculty")]
    fids = [r[0] for r in con.execute("SELECT id FROM faculty")]
    have = {(r[0], r[1]) for r in con.execute("SELECT student_id, course_id FROM marks")}
    con.close()
    os.remove(tmp)
    # (student, course) pairs with NO marks/attendance row yet — the only
    # pairs an INSERT into marks/attendance can validly use. The seeded
    # DB already holds ~70% of pairs, and marks/attendance share a
    # PRIMARY KEY (student_id, course_id), so inserting an existing pair
    # fails with a UNIQUE constraint error.
    missing = [(s, c) for s in sids for c in cids if (s, c) not in have]
    return students, sids, courses, cids, faculty, fids, missing


STUDENTS, SIDS, COURSES, CIDS, FACULTY, FIDS, MISSING_PAIRS = load_reference()
BRANCHES = ["CSE", "ECE", "ME", "CE"]
DEPTS = ["CSE", "ECE", "ME", "CE"]
_new_counter = [0]
_new_course_counter = [8]


def new_student(rng):
    _new_counter[0] += 1
    first = rng.choice(NEW_FIRST)
    last = rng.choice(NEW_LAST)
    name = f"{first} {last}"
    return {
        "name": name,
        "branch": rng.choice(BRANCHES),
        "year": rng.randint(1, 4),
        "email": f"{first.lower()}.{last.lower()}{_new_counter[0]}@aisat.ac.in",
        "id": 50 + _new_counter[0],
    }


def new_course(rng):
    _new_course_counter[0] += 1
    return {
        "name": rng.choice(NEW_COURSES),
        "credits": rng.choice([2, 3, 4]),
        "faculty_id": rng.choice(FIDS),
        "id": _new_course_counter[0],
    }


def slot(rng, kind):
    if kind == "student":
        return rng.choice(STUDENTS)
    if kind == "sid":
        return rng.choice(SIDS)
    if kind == "course":
        return rng.choice(COURSES)
    if kind == "cid":
        return rng.choice(CIDS)
    if kind == "faculty":
        return rng.choice(FACULTY)
    if kind == "fid":
        return rng.choice(FIDS)
    if kind == "branch":
        return rng.choice(BRANCHES)
    if kind == "dept":
        return rng.choice(DEPTS)
    if kind == "year":
        return rng.randint(1, 4)
    if kind == "marks":
        return rng.randint(40, 95)
    if kind == "marks2":
        return rng.randint(40, 95)
    if kind == "markval":
        return rng.randint(40, 98)
    if kind == "att":
        return rng.randint(55, 90)
    if kind == "att2":
        return rng.randint(55, 90)
    if kind == "attval":
        return rng.randint(60, 99)
    if kind == "n":
        return rng.randint(2, 15)
    if kind == "credits":
        return rng.choice([2, 3, 4])
    if kind == "freepair":
        return rng.choice(MISSING_PAIRS)
    if kind == "new":
        return new_student(rng)
    if kind == "newcourse":
        return new_course(rng)
    if kind == "newemail":
        first = rng.choice(NEW_FIRST).lower()
        last = rng.choice(NEW_LAST).lower()
        return f"{first}.{last}{rng.randint(1, 99)}@aisat.ac.in"
    raise ValueError(kind)


# Each template: (op, slot-kinds, [input phrasings], sql template)
# Placeholders use the slot names; "new" / "newcourse" are dicts.
TEMPLATES = [
    # ---- SELECT: students by branch / year ----
    ("SELECT", ["branch"], [
        "show {branch} students", "list {branch} students",
        "give me {branch} students", "show all students in {branch}",
        "students of {branch} branch",
    ], "SELECT * FROM students WHERE branch = '{branch}';"),
    ("SELECT", ["branch", "year"], [
        "show {branch} students in year {year}",
        "list year {year} {branch} students",
        "students in {branch} year {year}",
    ], "SELECT * FROM students WHERE branch = '{branch}' AND year = {year};"),
    ("SELECT", ["year"], [
        "show year {year} students", "list all students in year {year}",
        "students of year {year}",
    ], "SELECT * FROM students WHERE year = {year};"),
    ("SELECT", ["student"], [
        "show details of {student}", "find student {student}",
        "who is {student}", "get {student} details",
    ], "SELECT * FROM students WHERE name = '{student}';"),
    ("SELECT", ["branch"], [
        "how many students in {branch}", "count students in {branch}",
        "number of {branch} students",
    ], "SELECT COUNT(*) FROM students WHERE branch = '{branch}';"),
    ("SELECT", [], [
        "how many students are there", "total number of students",
        "count all students",
    ], "SELECT COUNT(*) FROM students;"),
    ("SELECT", [], [
        "how many students in each branch",
        "count students per branch", "students per branch",
    ], "SELECT branch, COUNT(*) FROM students GROUP BY branch;"),
    ("SELECT", [], [
        "how many students in each year", "count students per year",
    ], "SELECT year, COUNT(*) FROM students GROUP BY year;"),
    ("SELECT", ["branch"], [
        "show student names in {branch}", "list names of {branch} students",
        "names of students in {branch}",
    ], "SELECT name FROM students WHERE branch = '{branch}';"),
    ("SELECT", ["branch"], [
        "show emails of {branch} students", "list emails of {branch} students",
    ], "SELECT name, email FROM students WHERE branch = '{branch}';"),
    ("SELECT", ["branch", "year"], [
        "show emails of {branch} year {year} students",
        "list emails of year {year} {branch} students",
    ], "SELECT name, email FROM students WHERE branch = '{branch}' AND year = {year};"),
    ("SELECT", ["branch"], [
        "show {branch} students ordered by name",
        "list {branch} students sorted by name",
    ], "SELECT * FROM students WHERE branch = '{branch}' ORDER BY name;"),
    ("SELECT", ["branch"], [
        "show {branch} students ordered by year",
        "list {branch} students by year",
    ], "SELECT * FROM students WHERE branch = '{branch}' ORDER BY year;"),
    ("SELECT", [], [
        "list all students ordered by name", "show all students sorted by name",
    ], "SELECT * FROM students ORDER BY name;"),
    ("SELECT", ["student"], [
        "show email of {student}", "email address of {student}",
        "what is the email of {student}",
    ], "SELECT email FROM students WHERE name = '{student}';"),
    ("SELECT", ["branch", "year"], [
        "how many students in {branch} year {year}",
        "count {branch} students in year {year}",
    ], "SELECT COUNT(*) FROM students WHERE branch = '{branch}' AND year = {year};"),
    # ---- SELECT: marks ----
    ("SELECT", ["marks"], [
        "show students with marks above {marks}",
        "list students who scored more than {marks}",
        "students with marks greater than {marks}",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id WHERE marks.marks > {marks} "
       "ORDER BY marks.marks DESC;"),
    ("SELECT", ["marks"], [
        "show students with marks below {marks}",
        "list students who scored less than {marks}",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id WHERE marks.marks < {marks} "
       "ORDER BY marks.marks ASC;"),
    ("SELECT", ["branch", "marks"], [
        "show {branch} students with marks above {marks}",
        "list {branch} students who scored more than {marks}",
        "give me {branch} toppers above {marks}",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id WHERE students.branch = '{branch}' "
       "AND marks.marks > {marks} ORDER BY marks.marks DESC;"),
    ("SELECT", ["branch", "marks"], [
        "show {branch} students with marks below {marks}",
        "list {branch} students who scored less than {marks}",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id WHERE students.branch = '{branch}' "
       "AND marks.marks < {marks} ORDER BY marks.marks ASC;"),
    ("SELECT", ["marks", "marks2"], [
        "show students with marks between {marks} and {marks2}",
        "list students scoring between {marks} and {marks2}",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id WHERE marks.marks BETWEEN {marks} "
       "AND {marks2} ORDER BY marks.marks DESC;"),
    ("SELECT", ["branch", "year", "marks"], [
        "show {branch} year {year} students with marks above {marks}",
        "list year {year} {branch} students scoring more than {marks}",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id WHERE students.branch = '{branch}' "
       "AND students.year = {year} AND marks.marks > {marks} "
       "ORDER BY marks.marks DESC;"),
    ("SELECT", ["year", "marks"], [
        "show year {year} students with marks above {marks}",
        "list year {year} students scoring more than {marks}",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id WHERE students.year = {year} "
       "AND marks.marks > {marks} ORDER BY marks.marks DESC;"),
    ("SELECT", ["course", "marks"], [
        "show students in {course} with marks above {marks}",
        "list students in {course} who scored more than {marks}",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id JOIN courses "
       "ON marks.course_id = courses.id WHERE courses.name = '{course}' "
       "AND marks.marks > {marks} ORDER BY marks.marks DESC;"),
    ("SELECT", ["course", "marks"], [
        "show students in {course} with marks below {marks}",
        "list students in {course} who scored less than {marks}",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id JOIN courses "
       "ON marks.course_id = courses.id WHERE courses.name = '{course}' "
       "AND marks.marks < {marks} ORDER BY marks.marks ASC;"),
    ("SELECT", ["n"], [
        "top {n} students by marks", "show top {n} students",
        "best {n} students by marks",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id ORDER BY marks.marks DESC "
       "LIMIT {n};"),
    ("SELECT", ["n"], [
        "show {n} students with lowest marks",
        "bottom {n} students by marks",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id ORDER BY marks.marks ASC "
       "LIMIT {n};"),
    ("SELECT", ["course"], [
        "average marks in {course}", "what is the average marks in {course}",
        "avg marks for {course}",
    ], "SELECT AVG(marks.marks) FROM marks JOIN courses "
       "ON marks.course_id = courses.id WHERE courses.name = '{course}';"),
    ("SELECT", [], [
        "average marks per branch", "avg marks for each branch",
        "average marks by branch",
    ], "SELECT students.branch, AVG(marks.marks) FROM students JOIN marks "
       "ON students.id = marks.student_id GROUP BY students.branch;"),
    ("SELECT", ["year"], [
        "show average marks of year {year} students",
        "avg marks for year {year}",
    ], "SELECT AVG(marks.marks) FROM marks JOIN students "
       "ON marks.student_id = students.id WHERE students.year = {year};"),
    ("SELECT", ["course"], [
        "show marks of students in {course}",
        "list marks for {course}", "marks in {course}",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id JOIN courses "
       "ON marks.course_id = courses.id WHERE courses.name = '{course}';"),
    ("SELECT", ["student"], [
        "show marks of {student}", "marks of {student}",
        "what did {student} score",
    ], "SELECT courses.name, marks.marks FROM marks JOIN courses "
       "ON marks.course_id = courses.id JOIN students "
       "ON marks.student_id = students.id WHERE students.name = '{student}';"),
    ("SELECT", ["student"], [
        "average marks of {student}", "avg marks of {student}",
    ], "SELECT AVG(marks.marks) FROM marks JOIN students "
       "ON marks.student_id = students.id WHERE students.name = '{student}';"),
    ("SELECT", ["course"], [
        "highest marks in {course}", "maximum marks in {course}",
        "topper marks in {course}",
    ], "SELECT MAX(marks.marks) FROM marks JOIN courses "
       "ON marks.course_id = courses.id WHERE courses.name = '{course}';"),
    ("SELECT", ["course"], [
        "lowest marks in {course}", "minimum marks in {course}",
    ], "SELECT MIN(marks.marks) FROM marks JOIN courses "
       "ON marks.course_id = courses.id WHERE courses.name = '{course}';"),
    ("SELECT", ["course"], [
        "how many students took {course}", "count students in {course}",
    ], "SELECT COUNT(*) FROM marks JOIN courses "
       "ON marks.course_id = courses.id WHERE courses.name = '{course}';"),
    ("SELECT", ["course", "n"], [
        "top {n} students in {course} by marks",
        "best {n} students in {course}",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id JOIN courses "
       "ON marks.course_id = courses.id WHERE courses.name = '{course}' "
       "ORDER BY marks.marks DESC LIMIT {n};"),
    ("SELECT", ["student", "course"], [
        "show marks of {student} in {course}",
        "what did {student} score in {course}",
    ], "SELECT marks.marks FROM marks JOIN students "
       "ON marks.student_id = students.id JOIN courses "
       "ON marks.course_id = courses.id WHERE students.name = '{student}' "
       "AND courses.name = '{course}';"),
    ("SELECT", ["student"], [
        "show courses taken by {student}", "list courses of {student}",
        "which courses did {student} take",
    ], "SELECT courses.name FROM marks JOIN courses "
       "ON marks.course_id = courses.id JOIN students "
       "ON marks.student_id = students.id WHERE students.name = '{student}';"),
    ("SELECT", ["course"], [
        "show students in {course} ordered by marks",
        "list students in {course} by marks",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id JOIN courses "
       "ON marks.course_id = courses.id WHERE courses.name = '{course}' "
       "ORDER BY marks.marks DESC;"),
    # ---- SELECT: attendance ----
    ("SELECT", ["att"], [
        "show students with attendance below {att}",
        "list students with attendance less than {att}",
        "students with low attendance below {att}",
    ], "SELECT students.name, attendance.percent FROM students JOIN attendance "
       "ON students.id = attendance.student_id WHERE attendance.percent < {att};"),
    ("SELECT", ["att", "att2"], [
        "show students with attendance between {att} and {att2}",
        "list students with attendance from {att} to {att2}",
    ], "SELECT students.name, attendance.percent FROM students JOIN attendance "
       "ON students.id = attendance.student_id WHERE attendance.percent "
       "BETWEEN {att} AND {att2};"),
    ("SELECT", ["course", "att"], [
        "show students with attendance below {att} in {course}",
        "list students in {course} with attendance under {att}",
    ], "SELECT students.name, attendance.percent FROM students JOIN attendance "
       "ON students.id = attendance.student_id JOIN courses "
       "ON attendance.course_id = courses.id WHERE courses.name = '{course}' "
       "AND attendance.percent < {att};"),
    ("SELECT", ["student"], [
        "show attendance of {student}", "attendance of {student}",
        "what is the attendance of {student}",
    ], "SELECT courses.name, attendance.percent FROM attendance JOIN courses "
       "ON attendance.course_id = courses.id JOIN students "
       "ON attendance.student_id = students.id "
       "WHERE students.name = '{student}';"),
    ("SELECT", ["course"], [
        "average attendance in {course}", "avg attendance for {course}",
    ], "SELECT AVG(attendance.percent) FROM attendance JOIN courses "
       "ON attendance.course_id = courses.id WHERE courses.name = '{course}';"),
    ("SELECT", ["branch", "att"], [
        "show {branch} students with attendance below {att}",
        "list {branch} students with low attendance under {att}",
    ], "SELECT students.name, attendance.percent FROM students JOIN attendance "
       "ON students.id = attendance.student_id WHERE students.branch = '{branch}' "
       "AND attendance.percent < {att};"),
    ("SELECT", ["branch"], [
        "average attendance of {branch} students",
        "avg attendance for {branch} branch",
    ], "SELECT AVG(attendance.percent) FROM attendance JOIN students "
       "ON attendance.student_id = students.id WHERE students.branch = '{branch}';"),
    ("SELECT", ["course"], [
        "show attendance of students in {course}",
        "list attendance for {course}",
    ], "SELECT students.name, attendance.percent FROM students JOIN attendance "
       "ON students.id = attendance.student_id JOIN courses "
       "ON attendance.course_id = courses.id WHERE courses.name = '{course}';"),
    ("SELECT", ["student", "att"], [
        "show courses where {student} has attendance below {att}",
        "in which courses does {student} have attendance under {att}",
    ], "SELECT courses.name FROM attendance JOIN courses "
       "ON attendance.course_id = courses.id JOIN students "
       "ON attendance.student_id = students.id WHERE students.name = '{student}' "
       "AND attendance.percent < {att};"),
    ("SELECT", ["student", "course"], [
        "show attendance of {student} in {course}",
        "what is the attendance of {student} in {course}",
    ], "SELECT attendance.percent FROM attendance JOIN students "
       "ON attendance.student_id = students.id JOIN courses "
       "ON attendance.course_id = courses.id WHERE students.name = '{student}' "
       "AND courses.name = '{course}';"),
    # ---- SELECT: courses / faculty ----
    ("SELECT", [], [
        "list all courses", "show all courses", "what courses are there",
    ], "SELECT * FROM courses;"),
    ("SELECT", ["credits"], [
        "show courses with at least {credits} credits",
        "list courses with credits {credits} or more",
    ], "SELECT * FROM courses WHERE credits >= {credits};"),
    ("SELECT", ["credits"], [
        "show courses with {credits} credits",
        "list courses with exactly {credits} credits",
    ], "SELECT * FROM courses WHERE credits = {credits};"),
    ("SELECT", [], [
        "show total credits of all courses", "sum of credits of all courses",
    ], "SELECT SUM(credits) FROM courses;"),
    ("SELECT", [], [
        "list all faculty", "show all faculty members", "who are the faculty",
    ], "SELECT * FROM faculty;"),
    ("SELECT", ["dept"], [
        "show faculty in {dept}", "list faculty of {dept} department",
        "faculty members in {dept}",
    ], "SELECT * FROM faculty WHERE department = '{dept}';"),
    ("SELECT", ["faculty"], [
        "which courses does {faculty} teach",
        "show courses taught by {faculty}",
        "courses of {faculty}",
    ], "SELECT courses.name FROM courses JOIN faculty "
       "ON courses.faculty_id = faculty.id WHERE faculty.name = '{faculty}';"),
    ("SELECT", ["course"], [
        "who teaches {course}", "faculty for {course}",
        "show faculty of {course}",
    ], "SELECT faculty.name FROM faculty JOIN courses "
       "ON courses.faculty_id = faculty.id WHERE courses.name = '{course}';"),
    ("SELECT", ["dept"], [
        "how many faculty in {dept}", "count faculty in {dept}",
    ], "SELECT COUNT(*) FROM faculty WHERE department = '{dept}';"),
    ("SELECT", ["faculty"], [
        "show department of {faculty}", "which department is {faculty} in",
    ], "SELECT department FROM faculty WHERE name = '{faculty}';"),
    ("SELECT", ["dept"], [
        "show courses taught by faculty in {dept}",
        "list courses of {dept} faculty",
    ], "SELECT courses.name FROM courses JOIN faculty "
       "ON courses.faculty_id = faculty.id WHERE faculty.department = '{dept}';"),
    ("SELECT", ["course"], [
        "show students in {course}", "list students enrolled in {course}",
        "students taking {course}",
    ], "SELECT students.name FROM students JOIN marks "
       "ON students.id = marks.student_id JOIN courses "
       "ON marks.course_id = courses.id WHERE courses.name = '{course}';"),
    # ---- SELECT: combined ----
    ("SELECT", ["branch", "marks", "att"], [
        "show {branch} students with marks above {marks} and attendance below {att}",
        "list {branch} students scoring above {marks} with attendance under {att}",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id WHERE students.branch = '{branch}' "
       "AND marks.marks > {marks} AND students.id IN "
       "(SELECT student_id FROM attendance WHERE percent < {att});"),
    ("SELECT", ["branch", "marks", "att"], [
        "show {branch} students with marks above {marks} and attendance below {att} with percent",
        "list {branch} toppers above {marks} with low attendance under {att}",
    ], "SELECT students.name, marks.marks, attendance.percent FROM students "
       "JOIN marks ON students.id = marks.student_id JOIN attendance "
       "ON students.id = attendance.student_id WHERE students.branch = '{branch}' "
       "AND marks.marks > {marks} AND attendance.percent < {att};"),
    ("SELECT", ["n", "branch"], [
        "top {n} students in {branch} by marks",
        "best {n} {branch} students",
    ], "SELECT students.name, marks.marks FROM students JOIN marks "
       "ON students.id = marks.student_id WHERE students.branch = '{branch}' "
       "ORDER BY marks.marks DESC LIMIT {n};"),
    ("SELECT", [], [
        "show all marks", "list all marks records",
    ], "SELECT * FROM marks;"),
    # ---- INSERT ----
    ("INSERT", ["new"], [
        "add a new student {new[name]} in {new[branch]} year {new[year]}",
        "insert student {new[name]} {new[branch]} year {new[year]}",
        "add student named {new[name]} to {new[branch]}",
        "create a new student {new[name]} in {new[branch]}",
    ], "INSERT INTO students (id, name, branch, year, email) VALUES "
       "({new[id]}, '{new[name]}', '{new[branch]}', {new[year]}, '{new[email]}');"),
    ("INSERT", ["new"], [
        "add a new faculty member {new[name]} in {new[branch]} department",
        "insert faculty {new[name]} department {new[branch]}",
    ], "INSERT INTO faculty (id, name, department) VALUES "
       "({new[id]}, '{new[name]}', '{new[branch]}');"),
    ("INSERT", ["newcourse"], [
        "add a new course {newcourse[name]} with {newcourse[credits]} credits",
        "insert course {newcourse[name]} credits {newcourse[credits]}",
        "create a new course {newcourse[name]}",
    ], "INSERT INTO courses (id, name, credits, faculty_id) VALUES "
       "({newcourse[id]}, '{newcourse[name]}', {newcourse[credits]}, {newcourse[faculty_id]});"),
    ("INSERT", ["freepair", "markval"], [
        "add marks {markval} for student {freepair[0]} in course {freepair[1]}",
        "insert marks {markval} for student id {freepair[0]} course id {freepair[1]}",
        "record marks {markval} for student {freepair[0]} in course {freepair[1]}",
    ], "INSERT INTO marks (student_id, course_id, marks) VALUES "
       "({freepair[0]}, {freepair[1]}, {markval});"),
    ("INSERT", ["freepair", "attval"], [
        "add attendance {attval} for student {freepair[0]} in course {freepair[1]}",
        "insert attendance {attval} percent for student id {freepair[0]} course id {freepair[1]}",
    ], "INSERT INTO attendance (student_id, course_id, percent) VALUES "
       "({freepair[0]}, {freepair[1]}, {attval});"),
    # ---- UPDATE ----
    ("UPDATE", ["sid", "cid", "markval"], [
        "update marks of student {sid} in course {cid} to {markval}",
        "change marks for student {sid} course {cid} to {markval}",
        "set marks {markval} for student {sid} in course {cid}",
    ], "UPDATE marks SET marks = {markval} WHERE student_id = {sid} "
       "AND course_id = {cid};"),
    ("UPDATE", ["sid", "cid", "attval"], [
        "update attendance of student {sid} in course {cid} to {attval}",
        "change attendance for student {sid} course {cid} to {attval} percent",
        "set attendance {attval} for student {sid} in course {cid}",
    ], "UPDATE attendance SET percent = {attval} WHERE student_id = {sid} "
       "AND course_id = {cid};"),
    ("UPDATE", ["student", "branch"], [
        "change branch of {student} to {branch}",
        "update branch of {student} to {branch}",
        "move {student} to {branch} branch",
    ], "UPDATE students SET branch = '{branch}' WHERE name = '{student}';"),
    ("UPDATE", ["student", "year"], [
        "update year of {student} to {year}",
        "change year of {student} to {year}",
    ], "UPDATE students SET year = {year} WHERE name = '{student}';"),
    ("UPDATE", ["student", "newemail"], [
        "update email of {student} to {newemail}",
        "change email of {student} to {newemail}",
    ], "UPDATE students SET email = '{newemail}' WHERE name = '{student}';"),
    ("UPDATE", ["cid", "credits"], [
        "update credits of course {cid} to {credits}",
        "change credits for course id {cid} to {credits}",
    ], "UPDATE courses SET credits = {credits} WHERE id = {cid};"),
    ("UPDATE", ["cid", "fid"], [
        "change faculty of course {cid} to faculty {fid}",
        "update faculty of course id {cid} to {fid}",
    ], "UPDATE courses SET faculty_id = {fid} WHERE id = {cid};"),
    ("UPDATE", ["faculty", "dept"], [
        "change department of {faculty} to {dept}",
        "update department of {faculty} to {dept}",
    ], "UPDATE faculty SET department = '{dept}' WHERE name = '{faculty}';"),
    # ---- DELETE ----
    ("DELETE", ["student"], [
        "delete student {student}", "remove student {student}",
        "delete {student} from students",
    ], "DELETE FROM students WHERE name = '{student}';"),
    ("DELETE", ["sid"], [
        "delete student with id {sid}", "remove student id {sid}",
    ], "DELETE FROM students WHERE id = {sid};"),
    ("DELETE", ["branch", "year"], [
        "delete {branch} students in year {year}",
        "remove year {year} {branch} students",
    ], "DELETE FROM students WHERE branch = '{branch}' AND year = {year};"),
    ("DELETE", ["sid", "cid"], [
        "delete marks of student {sid} in course {cid}",
        "remove marks record for student {sid} course {cid}",
    ], "DELETE FROM marks WHERE student_id = {sid} AND course_id = {cid};"),
    ("DELETE", ["marks"], [
        "delete marks records below {marks}",
        "remove marks below {marks}",
    ], "DELETE FROM marks WHERE marks < {marks};"),
    ("DELETE", ["sid", "cid"], [
        "delete attendance of student {sid} in course {cid}",
        "remove attendance record for student {sid} course {cid}",
    ], "DELETE FROM attendance WHERE student_id = {sid} AND course_id = {cid};"),
    ("DELETE", ["faculty"], [
        "delete faculty {faculty}", "remove faculty member {faculty}",
    ], "DELETE FROM faculty WHERE name = '{faculty}';"),
    ("DELETE", ["course"], [
        "delete course {course}", "remove course {course}",
    ], "DELETE FROM courses WHERE name = '{course}';"),
]

# How many examples to draw from each operation.
TARGETS = {"SELECT": 12000, "INSERT": 1500, "UPDATE": 1500, "DELETE": 1000}
MAX_ATTEMPTS_PER_OP = 400000


def _draw(t, rng):
    """Draw one (input, sql) pair from a template, or None on error."""
    _, kinds, inputs, sql_t = t
    slots = {}
    for k in kinds:
        slots[k] = slot(rng, k)
    # BETWEEN needs low <= high, otherwise the SQL is nonsense.
    for lo, hi in (("marks", "marks2"), ("att", "att2")):
        if lo in slots and hi in slots and slots[lo] > slots[hi]:
            slots[lo], slots[hi] = slots[hi], slots[lo]
    flat = dict(slots)
    if "new" in slots:
        for kk, vv in slots["new"].items():
            flat[f"new[{kk}]"] = vv
    if "newcourse" in slots:
        for kk, vv in slots["newcourse"].items():
            flat[f"newcourse[{kk}]"] = vv
    # "freepair" is a (student_id, course_id) tuple: expand to indices.
    if "freepair" in slots:
        for i, vv in enumerate(slots["freepair"]):
            flat[f"freepair[{i}]"] = vv
    try:
        inp = rng.choice(inputs).format(**flat)
        sql = sql_t.format(**flat)
    except (KeyError, IndexError):
        return None
    return inp, sql


def generate():
    """Sample templates at random until every target is reached.

    Fixed per-template quotas cannot reach the targets: low-slot
    templates (e.g. branch-only) saturate after a few dozen unique
    examples. Random sampling lets high-cardinality templates
    (student+course, branch+marks) carry the count.
    """
    rng = random.Random(42)
    by_op = {op: [] for op in TARGETS}
    seen = set()
    # (student, course) pairs already used by an INSERT into each table:
    # a second INSERT of the same pair would fail the PK, so skip it.
    # (~125 free pairs per table; the other INSERT templates carry the
    # 1,500 target.)
    used_pairs = {"marks": set(), "attendance": set()}
    templates_by_op = {}
    for t in TEMPLATES:
        templates_by_op.setdefault(t[0], []).append(t)
    for op, target in TARGETS.items():
        attempts = 0
        while len(by_op[op]) < target and attempts < MAX_ATTEMPTS_PER_OP:
            attempts += 1
            drawn = _draw(rng.choice(templates_by_op[op]), rng)
            if drawn is None:
                continue
            inp, sql = drawn
            key = (inp.lower(), sql)
            if key in seen:
                continue
            m = re.match(
                r"INSERT INTO (marks|attendance) "
                r"\(student_id, course_id, (?:marks|percent)\) "
                r"VALUES \((\d+), (\d+),",
                sql,
            )
            if m:
                pair = (int(m.group(2)), int(m.group(3)))
                if pair in used_pairs[m.group(1)]:
                    continue
                used_pairs[m.group(1)].add(pair)
            seen.add(key)
            by_op[op].append({"input": inp, "sql": sql, "op": op})
        if len(by_op[op]) < target:
            print(f"WARNING: {op} reached only {len(by_op[op])}/{target} "
                  f"after {attempts} attempts (template space exhausted)")
    all_rows = []
    for op in ("SELECT", "INSERT", "UPDATE", "DELETE"):
        all_rows.extend(by_op[op])
    rng.shuffle(all_rows)
    return all_rows, by_op


def tokenize(text):
    return re.findall(r"[A-Za-z_]+|[0-9]+|[^\sA-Za-z0-9_]", text)


def build_vocab(rows):
    from collections import Counter
    c = Counter()
    for r in rows:
        c.update(tokenize(r["input"].lower()))
        c.update(tokenize(r["sql"]))
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for tok, _ in c.most_common():
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def main():
    rows, by_op = generate()
    print("counts:", {op: len(v) for op, v in by_op.items()},
          "total:", len(rows))
    rng = random.Random(42)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]
    assert len(train) + len(val) + len(test) == n, "split must cover all rows"
    os.makedirs(DATA, exist_ok=True)
    for name, split in (("dataset", rows), ("train", train),
                        ("val", val), ("test", test)):
        with open(os.path.join(DATA, f"{name}.jsonl"), "w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")
    vocab = build_vocab(rows)
    with open(os.path.join(DATA, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=1)
    print("splits: train", len(train), "val", len(val), "test", len(test),
          "| vocab size:", len(vocab))
    print("sample SELECT:", rows[0])
    print("sample write:",
          next(r for r in rows if r["op"] != "SELECT"))


if __name__ == "__main__":
    main()
