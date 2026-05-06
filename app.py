from flask import Flask, render_template, request, jsonify
import sqlite3, os, json, base64, re, requests
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# -------------------------------------------------------------------------
# GEMINI API Key (Loaded from .env)
# -------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

BASE_DIR   = os.path.dirname(__file__)
DB_PATH    = os.path.join(BASE_DIR, 'shokudou.db')
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'gif', 'webp', 'pdf'}

# -------------------------------------------------------------------------
# Database
# -------------------------------------------------------------------------
def get_db():
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2 is required for PostgreSQL but not installed. Install with 'pip install psycopg2-binary'")
        # Use PostgreSQL (Production/Supabase)
        # Handle Render's postgres:// vs postgresql:// issue
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        return conn
    else:
        # Use SQLite (Local)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def execute_db(conn, query, params=()):
    """Wrapper to handle differences between SQLite (?) and PostgreSQL (%s)"""
    is_pg = HAS_PSYCOPG2 and not isinstance(conn, sqlite3.Connection)
    if is_pg:
        query = query.replace('?', '%s')
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, params)
        return cur
    else:
        return conn.execute(query, params)

def init_db():
    conn = get_db()
    is_pg = HAS_PSYCOPG2 and not isinstance(conn, sqlite3.Connection)
    
    # Text types and syntax are slightly different but compatible for basic usage
    sql_menu = '''
        CREATE TABLE IF NOT EXISTS menu_data (
            id         SERIAL PRIMARY KEY if_not_exists_placeholder,
            date       TEXT    NOT NULL,
            meal_type  TEXT    NOT NULL,
            main_dish  TEXT    DEFAULT '',
            items      TEXT    DEFAULT '[]',
            kcal       TEXT    DEFAULT '',
            updated_at TEXT    DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, meal_type)
        )
    '''
    # Adjust for SQLite/Postgres differences in SERIAL/AUTOINCREMENT
    if is_pg:
        sql_menu = sql_menu.replace("id         SERIAL PRIMARY KEY if_not_exists_placeholder", "id SERIAL PRIMARY KEY")
        sql_sync = "CREATE TABLE IF NOT EXISTS sync_logs (key TEXT PRIMARY KEY, last_sync TEXT)"
    else:
        sql_menu = sql_menu.replace("id         SERIAL PRIMARY KEY if_not_exists_placeholder", "id INTEGER PRIMARY KEY AUTOINCREMENT")
        sql_sync = "CREATE TABLE IF NOT EXISTS sync_logs (key TEXT PRIMARY KEY, last_sync TEXT)"

    try:
        if is_pg:
            cur = conn.cursor()
            cur.execute(sql_menu)
            cur.execute(sql_sync)
            conn.commit()
            cur.close()
        else:
            conn.execute(sql_menu)
            conn.execute(sql_sync)
            conn.commit()
    finally:
        conn.close()

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def get_monday(dt):
    # weekday() returns 0 for Monday
    return (dt - timedelta(days=dt.weekday())).date()

def parse_with_gemini(mime_type, data_b64):
    prompt = (
        "You are an expert data extractor. Extract the menu from this Japanese cafeteria (食堂) weekly menu image or pdf.\n"
        "Return ONLY a valid JSON object matching the exact format below, without any markdown formatting.\n"
        '{"days":[{"date":"2026-05-04",'
        '"朝食":{"main":"ライス / パン","items":["Item 1", "Item 2"],"kcal":"1100kcal"},'
        '"昼食":{"main":"","items":["Dish A"],"kcal":"900kcal"},'
        '"夕食":{"main":"","items":["Dish B"],"kcal":"950kcal"}}]}\n'
        "CRITICAL RULES:\n"
        "1. Columns represent days (月〜日). Extract dates from headers.\n"
        "2. 朝食=morning, 昼食=lunch, 夕食=dinner.\n"
        "3. Keep descriptions EXTREMELY short and concise. DO NOT write full sentences.\n"
        "4. DO NOT REPEAT ANY TEXT. If you find yourself repeating, stop immediately.\n"
        "5. IMPORTANT: Extract the original text in JAPANESE. Do not translate to other languages."
    )

    body = {
        'systemInstruction': {'parts': [{'text': 'You are a Japanese cafeteria menu parser. Always return ONLY valid JSON matching the requested schema.'}]},
        'contents': [{'parts': [
            {'text': prompt},
            {'inlineData': {'mimeType': mime_type, 'data': data_b64}}
        ]}],
        'generationConfig': {
            'temperature': 0.1, 
            'maxOutputTokens': 32768,
            'responseMimeType': 'application/json'
        }
    }

    # List of models to try in order
    models = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-flash-latest']
    
    last_error = ""
    for model in models:
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}'
        
        try:
            session = requests.Session()
            resp = session.post(url, json=body, timeout=60)
            
            # Detailed logging
            with open(os.path.join(BASE_DIR, 'debug_api.txt'), 'a', encoding='utf-8') as f:
                f.write(f"--- {datetime.now()} | Model: {model} ---\nStatus: {resp.status_code}\nResponse: {resp.text}\n---\n")
            
            if resp.status_code == 200:
                text = resp.json()['candidates'][0]['content']['parts'][0]['text']
                break # Success!
            elif resp.status_code == 429:
                last_error = "AI Quota Full (429)"
                continue # Try next model
            else:
                resp.raise_for_status()
        except Exception as e:
            last_error = str(e)
            continue
    else:
        # If all models failed
        raise Exception(f"AI Error: {last_error}. AI Quota exceeded (Gemini API 429). Please try again tomorrow or change the API Key in .env.")

    # Clean the response text (same as before)
    
    cleaned_text = text.strip()
    if cleaned_text.startswith('```json'):
        cleaned_text = cleaned_text[7:]
    elif cleaned_text.startswith('```'):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith('```'):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        # Fallback: if JSON is broken, try to find the first { and last }
        start = cleaned_text.find('{')
        end = cleaned_text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned_text[start:end+1])
            except:
                pass
        raise Exception(f"JSON Parsing Error: {str(e)} | Text Preview: {cleaned_text[:500]}...")

def sync_menu_from_school():
    """Calculates current week's Monday and fetches PDF if missing from DB."""
    now = datetime.now()
    monday = get_monday(now)
    monday_str = monday.strftime("%Y-%m-%d")
    
    with get_db() as conn:
        # 1. Check if data for this week already exists
        row = execute_db(conn, 'SELECT 1 FROM menu_data WHERE date = ? LIMIT 1', (monday_str,)).fetchone()
        if row: return

        # 2. Check Cooldown
        log = execute_db(conn, 'SELECT last_sync FROM sync_logs WHERE key = ?', (monday_str,)).fetchone()
        if log:
            last_time = datetime.fromisoformat(log[0] if isinstance(log, tuple) else log['last_sync'])
            if datetime.now() - last_time < timedelta(hours=1):
                return

        # Log sync attempt (Using ON CONFLICT instead of INSERT OR REPLACE)
        execute_db(conn, '''
            INSERT INTO sync_logs (key, last_sync) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET last_sync = excluded.last_sync
        ''', (monday_str, datetime.now().isoformat()))
        if hasattr(conn, 'commit'): conn.commit()

    # 3. Proceed with sync
    url_date = monday.strftime("%Y%m%d")
    pdf_url = f"https://www.kochi-ct.ac.jp/files/uploads/kondate{url_date}.pdf"
    
    try:
        resp = requests.get(pdf_url, timeout=30)
        if resp.status_code == 200:
            pdf_b64 = base64.b64encode(resp.content).decode()
            parsed = parse_with_gemini('application/pdf', pdf_b64)
            
            if parsed and 'days' in parsed:
                with get_db() as conn:
                    for day in parsed['days']:
                        d = day.get('date')
                        if not d: continue
                        for meal_type in ['朝食', '昼食', '夕食']:
                            info = day.get(meal_type)
                            if info:
                                items_json = json.dumps(info.get('items', []), ensure_ascii=False)
                                execute_db(conn, '''
                                    INSERT INTO menu_data (date, meal_type, main_dish, items, kcal, updated_at)
                                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                                    ON CONFLICT(date, meal_type) DO UPDATE SET
                                        main_dish  = excluded.main_dish,
                                        items      = excluded.items,
                                        kcal       = excluded.kcal,
                                        updated_at = CURRENT_TIMESTAMP
                                ''', (d, meal_type, info.get('main', ''), items_json, info.get('kcal', '')))
                    
                    execute_db(conn, '''
                        INSERT INTO sync_logs (key, last_sync) VALUES (?, ?)
                        ON CONFLICT(key) DO UPDATE SET last_sync = excluded.last_sync
                    ''', (f"{monday_str}_status", "success"))
                    if hasattr(conn, 'commit'): conn.commit()
                    print(f"Auto-sync successful for {monday_str}")
        else:
            with get_db() as conn:
                conn.execute('INSERT INTO sync_logs (date_monday, status, error_msg) VALUES (?, ?, ?)', 
                             (monday_str, 'error', f'HTTP {resp.status_code}'))
                conn.commit()
    except Exception as e:
        print(f"Auto-sync failed for {pdf_url}: {e}")
        with get_db() as conn:
            execute_db(conn, '''
                INSERT INTO sync_logs (key, last_sync) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET last_sync = excluded.last_sync
            ''', (f"{monday_str}_error", str(e)[:100]))
            if hasattr(conn, 'commit'): conn.commit()

# -------------------------------------------------------------------------
# Pages
# -------------------------------------------------------------------------
@app.route('/')
def menu_view():
    return render_template('menu.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

# -------------------------------------------------------------------------
# API
# -------------------------------------------------------------------------
@app.route('/api/menu')
def api_get_menu():
    # Attempt auto-fetch if current week is missing
    sync_menu_from_school()

    with get_db() as conn:
        rows = execute_db(conn, 
            'SELECT date, meal_type, main_dish, items, kcal FROM menu_data ORDER BY date'
        ).fetchall()
    result = {}
    for row in rows:
        d = row[0] if isinstance(row, tuple) else row['date']
        mtype = row[1] if isinstance(row, tuple) else row['meal_type']
        if d not in result:
            result[d] = {}
        
        # Determine values based on row type (Postgres RealDictCursor vs SQLite Row)
        main = row[2] if isinstance(row, tuple) else row['main_dish']
        items = row[3] if isinstance(row, tuple) else row['items']
        kcal = row[4] if isinstance(row, tuple) else row['kcal']

        result[d][mtype] = {
            'main':  main,
            'items': json.loads(items),
            'kcal':  kcal,
        }
    if not isinstance(conn, sqlite3.Connection): conn.close()
    return jsonify(result)

@app.route('/api/menu/save', methods=['POST'])
def api_save_menu():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    saved = 0
    with get_db() as conn:
        for date, meals in data.items():
            for meal_type, info in meals.items():
                items_json = json.dumps(info.get('items', []), ensure_ascii=False)
                execute_db(conn, '''
                    INSERT INTO menu_data (date, meal_type, main_dish, items, kcal, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(date, meal_type) DO UPDATE SET
                        main_dish  = excluded.main_dish,
                        items      = excluded.items,
                        kcal       = excluded.kcal,
                        updated_at = CURRENT_TIMESTAMP
                ''', (date, meal_type,
                      info.get('main', ''), items_json, info.get('kcal', '')))
                saved += 1
        if hasattr(conn, 'commit'): conn.commit()
    if not isinstance(conn, sqlite3.Connection): conn.close()
    return jsonify({'success': True, 'saved': saved})

@app.route('/api/menu/delete-day', methods=['POST'])
def api_delete_day():
    date = request.get_json().get('date')
    if not date:
        return jsonify({'error': 'No date provided'}), 400
    with get_db() as conn:
        execute_db(conn, 'DELETE FROM menu_data WHERE date = ?', (date,))
        if hasattr(conn, 'commit'): conn.commit()
    if not isinstance(conn, sqlite3.Connection): conn.close()
    return jsonify({'success': True})

@app.route('/api/menu/delete-all', methods=['POST'])
def api_delete_all():
    with get_db() as conn:
        execute_db(conn, 'DELETE FROM menu_data')
        if hasattr(conn, 'commit'): conn.commit()
    if not isinstance(conn, sqlite3.Connection): conn.close()
    return jsonify({'success': True})

@app.route('/api/parse-image', methods=['POST'])
def api_parse_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = secure_filename(f"menu_{ts}_{file.filename}")
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    with open(file_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    mime = file.content_type or 'image/jpeg'

    try:
        parsed = parse_with_gemini(mime, img_b64)
        return jsonify({
            'success': True,
            'data': parsed,
            'file_url': f'/static/uploads/{filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fetch-url-menu', methods=['POST'])
def api_fetch_url_menu():
    date_str = request.get_json().get('date')
    if not date_str:
        return jsonify({'error': 'No date provided'}), 400
        
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        url_date = dt.strftime("%Y%m%d")
        pdf_url = f"https://www.kochi-ct.ac.jp/files/uploads/kondate{url_date}.pdf"
        
        resp = requests.get(pdf_url, timeout=30)
        if resp.status_code == 404:
            return jsonify({'error': f'PDF file not found on the website (URL: {pdf_url})'}), 404
        resp.raise_for_status()
        
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"kondate_{url_date}_{ts}.pdf"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, 'wb') as f:
            f.write(resp.content)
            
        pdf_b64 = base64.b64encode(resp.content).decode()
        
        parsed = parse_with_gemini('application/pdf', pdf_b64)
        return jsonify({
            'success': True,
            'data': parsed,
            'file_url': f'/static/uploads/{filename}',
            'is_pdf': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
