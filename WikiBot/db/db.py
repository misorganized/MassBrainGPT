"""
Wrapper for using a SQLite3 database to store a list because I like making things harder to follow.
"""
import sqlite3


class db:
    def __init__(self, file):
        self.file = file

    def append(self, item):
        with sqlite3.connect(self.file) as db:
            c = db.cursor()
            c.execute("INSERT INTO list VALUES (?)", (item,))
            db.commit()

    def read(self):
        resp = None
        with sqlite3.connect(self.file) as db:
            c = db.cursor()
            r = c.execute("SELECT * FROM list", )
            resp = [x[0] for x in r.fetchall()]
            db.close()
        return resp


def create_db(name):
    with sqlite3.connect(name) as db:
        db.execute("CREATE TABLE list (i TEXT)")
        db.commit()


if __name__ == '__main__':
    name = input("Name of the database (including .db): ")
    create_db(name)