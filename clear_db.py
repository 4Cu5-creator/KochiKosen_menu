import sqlite3
conn = sqlite3.connect('shokudou.db')
conn.execute('DELETE FROM menu_data')
conn.execute('DELETE FROM sync_logs')
conn.commit()
print("Database cleared.")
conn.close()
