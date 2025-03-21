import pandas as pd
import sqlite3
from typing import List, Dict, Optional
import json
import os

class TableHandler:
    def __init__(self, db_path: str = "tables.db"):
        """Initialize the table handler with a SQLite database."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.setup_database()

    def setup_database(self):
        """Create necessary tables if they don't exist."""
        # Drop existing tables to reset schema
        self.cursor.execute("DROP TABLE IF EXISTS table_data")
        self.cursor.execute("DROP TABLE IF EXISTS tables")
        
        # Create tables table with metadata column
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS tables (
            table_name TEXT PRIMARY KEY,
            metadata TEXT,  -- JSON string of metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create table_data table to store actual table data
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT,
            row_data TEXT,  -- JSON string of row data
            FOREIGN KEY (table_name) REFERENCES tables (table_name)
        )
        ''')
        
        self.conn.commit()

    def store_table(self, df: pd.DataFrame, table_name: str, metadata: Dict = None) -> None:
        """Store a pandas DataFrame in the database."""
        try:
            # Convert metadata to string if it's a dict
            if isinstance(metadata, dict):
                metadata = json.dumps(metadata)
            
            # Store table metadata
            self.cursor.execute("""
                INSERT OR REPLACE INTO tables (table_name, metadata)
                VALUES (?, ?)
            """, (table_name, metadata))
            
            # Store table data
            for _, row in df.iterrows():
                row_data = json.dumps(row.to_dict())
                self.cursor.execute("""
                    INSERT INTO table_data (table_name, row_data)
                    VALUES (?, ?)
                """, (table_name, row_data))
            
            self.conn.commit()
            print(f"Successfully stored table: {table_name}")
        except Exception as e:
            print(f"Error storing table {table_name}: {e}")
            self.conn.rollback()

    def query_table(self, table_name: str) -> pd.DataFrame:
        """Query a table by name and return as DataFrame."""
        try:
            # Get all data for the table
            self.cursor.execute('SELECT row_data FROM table_data WHERE table_name = ?', (table_name,))
            rows = [json.loads(row[0]) for row in self.cursor.fetchall()]
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"Error querying table {table_name}: {e}")
            return pd.DataFrame()

    def get_table_metadata(self, table_name: str) -> Dict:
        """Get metadata for a specific table."""
        self.cursor.execute('SELECT metadata FROM tables WHERE table_name = ?', (table_name,))
        row = self.cursor.fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return {}

    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        self.cursor.execute('SELECT table_name FROM tables')
        return [row[0] for row in self.cursor.fetchall()]

    def close(self):
        """Close the database connection."""
        self.conn.close() 
