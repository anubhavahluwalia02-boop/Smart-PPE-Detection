"""
SQLite Database module for logging and querying PPE violations.
"""
import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import config

def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Creates a database connection with dict-like row access."""
    path = db_path or config.DATABASE_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Optional[Path] = None):
    """Initializes the database schema and indices."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            violation_type TEXT NOT NULL,
            missing_ppe TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            source TEXT NOT NULL,
            evidence_path TEXT,
            person_count INTEGER DEFAULT 1,
            notes TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON violations(timestamp DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_violation_type ON violations(violation_type)")
    conn.commit()
    conn.close()

def log_violation(
    violation_type: str,
    missing_ppe: str,
    confidence: float = 0.0,
    source: str = "webcam",
    evidence_path: Optional[str] = None,
    person_count: int = 1,
    notes: str = "",
    db_path: Optional[Path] = None
) -> int:
    """Inserts a new violation record into SQLite."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO violations 
        (timestamp, violation_type, missing_ppe, confidence, source, evidence_path, person_count, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_iso,
        violation_type,
        missing_ppe,
        round(float(confidence), 3),
        source,
        evidence_path or "",
        person_count,
        notes
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_violations(
    limit: int = 100,
    filter_type: Optional[str] = None,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Retrieves filtered violation records."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    query = "SELECT * FROM violations WHERE 1=1"
    params = []
    
    if filter_type and filter_type.strip() and filter_type != "ALL":
        query += " AND violation_type = ?"
        params.append(filter_type.strip())
        
    if search and search.strip():
        search_term = f"%{search.strip()}%"
        query += " AND (missing_ppe LIKE ? OR notes LIKE ? OR source LIKE ?)"
        params.extend([search_term, search_term, search_term])
        
    if start_date and start_date.strip():
        query += " AND timestamp >= ?"
        params.append(start_date.strip())
        
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_violation_by_id(record_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single violation record by ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM violations WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def delete_violation(record_id: int, db_path: Optional[Path] = None) -> bool:
    """Deletes a violation record."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM violations WHERE id = ?", (record_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted

def get_dashboard_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Computes summary metrics for the safety compliance dashboard.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM violations")
    total_violations = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM violations WHERE violation_type = 'HELMET_VIOLATION'")
    helmet_violations = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM violations WHERE violation_type = 'SAFETY_VEST_VIOLATION'")
    vest_violations = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM violations WHERE violation_type = 'MULTIPLE_VIOLATIONS'")
    multiple_violations = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM violations WHERE violation_type = 'NEEDS_REVIEW'")
    needs_review = cursor.fetchone()[0] or 0

    # Hourly distribution for recent chart (last 24h / recent entries)
    cursor.execute("""
        SELECT substr(timestamp, 12, 2) as hour, COUNT(*) as count 
        FROM violations 
        GROUP BY hour 
        ORDER BY hour ASC 
        LIMIT 24
    """)
    hourly_data = [dict(row) for row in cursor.fetchall()]
    
    # Recent 6 violations
    cursor.execute("SELECT * FROM violations ORDER BY id DESC LIMIT 6")
    recent_violations = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "total_violations": total_violations,
        "helmet_violations": helmet_violations,
        "vest_violations": vest_violations,
        "multiple_violations": multiple_violations,
        "needs_review": needs_review,
        "hourly_data": hourly_data,
        "recent_violations": recent_violations
    }

def clear_records(db_path: Optional[Path] = None):
    """Removes all violation records."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM violations")
    conn.commit()
    conn.close()
