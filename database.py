import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

# Assuming these exist from your Correlation/Scoring engines
# from models import Trail, DetectionEvent, ScoreResult

DB_PATH = "guardly_memory.db"

def get_connection() -> sqlite3.Connection:
    """Returns a configured SQLite connection optimized for edge hardware."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # WAL mode handles concurrent reads (FastAPI) and writes (Workers) beautifully
    conn.execute('pragma journal_mode=wal') 
    return conn

def init_db():
    """Initializes the schema if it doesn't exist. Called on startup."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trails (
                trail_id TEXT PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trail_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trail_id TEXT,
                camera_id TEXT,
                timestamp TIMESTAMP,
                is_restricted_zone BOOLEAN,
                FOREIGN KEY(trail_id) REFERENCES trails(trail_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trail_id TEXT,
                score REAL,
                factors_json TEXT,
                timestamp TIMESTAMP,
                is_acknowledged BOOLEAN DEFAULT 0,
                FOREIGN KEY(trail_id) REFERENCES trails(trail_id)
            )
        ''')
        
        # Index for the Scoring Engine to quickly count route history
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_route_lookup 
            ON trail_events(camera_id, timestamp)
        ''')
        
        conn.commit()

def save_trail_and_incident(trail, score_result=None):
    """
    Saves a completed trail and its events. 
    If score_result is an alert, logs the incident.
    """
    if not trail.events:
        return

    def format_ts(ts):
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts).isoformat()
        return ts.isoformat()

    start_time = format_ts(trail.events[0].timestamp)
    end_time = format_ts(trail.events[-1].timestamp)

    with get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT OR IGNORE INTO trails (trail_id, start_time, end_time) VALUES (?, ?, ?)',
            (trail.trail_id, start_time, end_time)
        )
        
        # 2. Save Events
        for event in trail.events:
            # FIX: Safely get the restricted zone boolean
            is_restricted = getattr(event, 'is_restricted_zone', False)
            
            cursor.execute('''
                INSERT INTO trail_events (trail_id, camera_id, timestamp, is_restricted_zone)
                VALUES (?, ?, ?, ?)
            ''', (trail.trail_id, event.camera_id, format_ts(event.timestamp), is_restricted))
            
        # 3. Save Incident (if applicable)
        if score_result and score_result.is_alert:
            factors_json = json.dumps(score_result.factors)
            cursor.execute('''
                INSERT INTO incidents (trail_id, score, factors_json, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (trail.trail_id, score_result.total_score, factors_json, end_time))
            
        conn.commit()

def get_route_seen_count(camera_sequence: List[str]) -> int:
    """
    Used by the Scoring Engine: How many times have we seen someone go 
    from Camera A to Camera B? 
    """
    if len(camera_sequence) < 2:
        return 0
        
    start_cam = camera_sequence[-2]
    end_cam = camera_sequence[-1]
    
    # We look for trails that contain start_cam followed by end_cam.
    # This raw SQL is highly optimized for SQLite.
    query = '''
        SELECT COUNT(DISTINCT t1.trail_id) 
        FROM trail_events t1
        JOIN trail_events t2 ON t1.trail_id = t2.trail_id
        WHERE t1.camera_id = ? 
          AND t2.camera_id = ?
          AND t1.timestamp < t2.timestamp
    '''
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (start_cam, end_cam))
        result = cursor.fetchone()
        return result[0] if result else 0

def get_active_incidents() -> List[dict]:
    """Used by FastAPI to feed the Guard Dashboard."""
    query = '''
        SELECT i.incident_id, i.trail_id, i.score, i.timestamp, t.camera_id as last_camera
        FROM incidents i
        JOIN trail_events t ON i.trail_id = t.trail_id
        WHERE i.is_acknowledged = 0
        GROUP BY i.incident_id
        ORDER BY i.timestamp DESC
    '''
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        # Convert sqlite3.Row objects to dicts for easy JSON serialization in FastAPI
        return [dict(row) for row in cursor.fetchall()]

def acknowledge_incident(incident_id: int):
    """Guard clicks 'Acknowledge' on the UI."""
    with get_connection() as conn:
        conn.execute('UPDATE incidents SET is_acknowledged = 1 WHERE incident_id = ?', (incident_id,))
        conn.commit()