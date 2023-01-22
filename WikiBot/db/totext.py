import sqlite3

# Connect to the database
conn = sqlite3.connect('../Wikipedia.db')
cursor = conn.cursor()

# Select all of the values from the 'list' table
cursor.execute("SELECT * FROM list")

# Write the values to a text file
with open("../Wikipedia.txt", "w", encoding='utf-8', errors='ignore') as f:
    for row in cursor:
        f.write(" ".join([str(i) for i in row]) + "\n")

# Close the cursor and connection
cursor.close()
conn.close()
