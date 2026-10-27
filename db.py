"""
SmartRail - SQLite Database Management Module
Handles persistent logging of alerts, station master decisions, civic events, and performance metrics.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smartrail.db")

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a connection with row factory enabled."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes tables if they do not exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        location TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        confidence REAL DEFAULT 0.90,
        risk_level TEXT NOT NULL,
        recommended_action TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        override_reason TEXT,
        completed INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        response_time_sec REAL DEFAULT 0.0
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        location TEXT NOT NULL,
        details TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id TEXT,
        action_type TEXT NOT NULL,
        operator_id TEXT NOT NULL,
        details TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        response_time_sec REAL DEFAULT 0.0
    );
    """)

    conn.commit()
    conn.close()

def log_alert(
    alert_id: str,
    alert_type: str,
    location: str,
    sim_timestamp: str,
    risk_level: str,
    recommended_action: str,
    confidence: float = 0.92,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """Logs a new alert if it does not already exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_iso = datetime.now().isoformat()
    try:
        cursor.execute("""
        INSERT OR IGNORE INTO alerts 
        (id, type, location, timestamp, confidence, risk_level, recommended_action, status, override_reason, completed, created_at, response_time_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, 0, ?, 0.0)
        """, (alert_id, alert_type, location, sim_timestamp, confidence, risk_level, recommended_action, now_iso))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"[SmartRail DB Error] Failed to log alert: {e}")
        return False
    finally:
        conn.close()

def update_alert_status(
    alert_id: str,
    status: str,
    operator_id: str = "StationMaster_01",
    override_reason: Optional[str] = None,
    completed: int = 0,
    response_time_sec: float = 5.0,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """Updates the decision status for an alert (approved, rejected, overridden)."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    sim_now = datetime.now().strftime("%H:%M:%S")
    try:
        cursor.execute("""
        UPDATE alerts 
        SET status = ?, override_reason = ?, completed = ?, response_time_sec = ?
        WHERE id = ?
        """, (status, override_reason, completed, response_time_sec, alert_id))
        
        # Log to actions table
        details = f"Alert {alert_id} marked as {status}."
        if override_reason:
            details += f" Reason: {override_reason}"
        cursor.execute("""
        INSERT INTO actions (alert_id, action_type, operator_id, details, timestamp, response_time_sec)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (alert_id, status, operator_id, details, sim_now, response_time_sec))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"[SmartRail DB Error] Failed to update alert status: {e}")
        return False
    finally:
        conn.close()

def mark_alert_completed(alert_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Sets completed flag to 1."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE alerts SET completed = 1 WHERE id = ?", (alert_id,))
        conn.commit()
        return True
    except Exception as e:
        print(f"[SmartRail DB Error] Failed to complete alert: {e}")
        return False
    finally:
        conn.close()

def get_alerts(status_filter: Optional[str] = None, limit: int = 100, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves list of alerts."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        if status_filter:
            cursor.execute("SELECT * FROM alerts WHERE status = ? ORDER BY id DESC LIMIT ?", (status_filter, limit))
        else:
            cursor.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"[SmartRail DB Error] Failed to get alerts: {e}")
        return []
    finally:
        conn.close()

def log_event(event_type: str, location: str, details: str, sim_timestamp: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Logs an operational, civic or vision event."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_iso = datetime.now().isoformat()
    try:
        cursor.execute("""
        INSERT INTO events (event_type, location, details, timestamp, created_at)
        VALUES (?, ?, ?, ?, ?)
        """, (event_type, location, details, sim_timestamp, now_iso))
        conn.commit()
        return True
    except Exception as e:
        print(f"[SmartRail DB Error] Failed to log event: {e}")
        return False
    finally:
        conn.close()

def get_events(limit: int = 100, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent event logs."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"[SmartRail DB Error] Failed to get events: {e}")
        return []
    finally:
        conn.close()

def get_metrics_summary(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Computes operator approval metrics."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT status, response_time_sec FROM alerts")
        rows = cursor.fetchall()
        total = len(rows)
        approved = sum(1 for r in rows if r['status'] == 'approved')
        rejected = sum(1 for r in rows if r['status'] == 'rejected')
        overridden = sum(1 for r in rows if r['status'] == 'overridden')
        pending = sum(1 for r in rows if r['status'] == 'pending')
        
        times = [r['response_time_sec'] for r in rows if r['status'] in ('approved', 'rejected', 'overridden') and r['response_time_sec'] > 0]
        avg_resp_time = round(sum(times) / len(times), 2) if times else 0.0
        override_rate = round((overridden / total * 100), 1) if total > 0 else 0.0

        return {
            "total_alerts": total,
            "approved": approved,
            "rejected": rejected,
            "overridden": overridden,
            "pending": pending,
            "override_rate_pct": override_rate,
            "avg_response_time_sec": avg_resp_time
        }
    except Exception as e:
        print(f"[SmartRail DB Error] Failed to compute metrics: {e}")
        return {
            "total_alerts": 0, "approved": 0, "rejected": 0,
            "overridden": 0, "pending": 0, "override_rate_pct": 0.0,
            "avg_response_time_sec": 0.0
        }
    finally:
        conn.close()

def reset_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Wipes tables for a fresh simulation run."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS alerts")
    cursor.execute("DROP TABLE IF EXISTS events")
    cursor.execute("DROP TABLE IF EXISTS actions")
    conn.commit()
    conn.close()
    init_db(db_path)
