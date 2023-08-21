import sqlite3

def connection_sqlite3():
    connection = sqlite3.connect("database\database.sqlite")

    cursor = connection.cursor()

    cursor.execute("SELECT * FROM Tweets")

    col_names = list(map(lambda x: x[0], cursor.description))

    base = cursor.fetchall()

    return base, col_names

if __name__ == "__main__":
    connection_sqlite3()
