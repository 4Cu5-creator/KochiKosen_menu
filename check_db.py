import sqlite3
conn = sqlite3.connect('shokudou.db')
print("Dates in menu_data:")
for row in conn.execute('SELECT DISTINCT date FROM menu_data ORDER BY date'):
    print(row[0])
print("\nSync logs:")
for row in conn.execute('SELECT * FROM sync_logs'):
    print(row)
conn.close()
