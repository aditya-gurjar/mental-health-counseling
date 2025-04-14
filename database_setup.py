import pandas as pd
import sqlite3
import json
from pathlib import Path
import os

def create_database_schema(db_path):
    """Create the SQLite database schema"""
    print(f"Creating database at {db_path}...")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to the database (will create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    print("Creating tables...")
    
    # Main conversations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        conversation_id INTEGER PRIMARY KEY,
        context TEXT NOT NULL,
        response TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Issues table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS issues (
        issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_name TEXT UNIQUE NOT NULL
    )
    ''')
    
    # Response types table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS response_types (
        response_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
        response_type_name TEXT UNIQUE NOT NULL
    )
    ''')
    
    # Therapeutic approaches table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS therapeutic_approaches (
        approach_id INTEGER PRIMARY KEY AUTOINCREMENT,
        approach_name TEXT UNIQUE NOT NULL
    )
    ''')
    
    # Junction table for conversations and issues (many-to-many)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversation_issues (
        conversation_id INTEGER,
        issue_id INTEGER,
        PRIMARY KEY (conversation_id, issue_id),
        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id),
        FOREIGN KEY (issue_id) REFERENCES issues (issue_id)
    )
    ''')
    
    # Junction table for conversations and response types (many-to-many)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversation_response_types (
        conversation_id INTEGER,
        response_type_id INTEGER,
        PRIMARY KEY (conversation_id, response_type_id),
        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id),
        FOREIGN KEY (response_type_id) REFERENCES response_types (response_type_id)
    )
    ''')
    
    # Junction table for conversations and therapeutic approaches (many-to-many)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversation_approaches (
        conversation_id INTEGER,
        approach_id INTEGER,
        PRIMARY KEY (conversation_id, approach_id),
        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id),
        FOREIGN KEY (approach_id) REFERENCES therapeutic_approaches (approach_id)
    )
    ''')
    
    # Create a search index table to store tokenized text for faster searching
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
        conversation_id, 
        text, 
        tokenize='porter'
    )
    ''')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Database schema created successfully.")
    return True

def populate_label_tables(conn, labeled_data):
    """Populate the label tables with unique values from the dataset"""
    cursor = conn.cursor()
    
    # Extract all unique issues
    all_issues = set()
    for issues_json in labeled_data['issues']:
        issues = json.loads(issues_json)
        all_issues.update(issues)
    
    # Extract all unique response types
    all_response_types = set()
    for types_json in labeled_data['response_types']:
        types = json.loads(types_json)
        all_response_types.update(types)
    
    # Extract all unique therapeutic approaches
    all_approaches = set()
    for approaches_json in labeled_data['therapeutic_approaches']:
        approaches = json.loads(approaches_json)
        all_approaches.update(approaches)
    
    # Insert unique issues
    print(f"Inserting {len(all_issues)} unique issues...")
    for issue in all_issues:
        cursor.execute("INSERT OR IGNORE INTO issues (issue_name) VALUES (?)", (issue,))
    
    # Insert unique response types
    print(f"Inserting {len(all_response_types)} unique response types...")
    for response_type in all_response_types:
        cursor.execute("INSERT OR IGNORE INTO response_types (response_type_name) VALUES (?)", (response_type,))
    
    # Insert unique therapeutic approaches
    print(f"Inserting {len(all_approaches)} unique therapeutic approaches...")
    for approach in all_approaches:
        cursor.execute("INSERT OR IGNORE INTO therapeutic_approaches (approach_name) VALUES (?)", (approach,))
    
    # Commit changes
    conn.commit()
    
    print("Label tables populated successfully.")
    return True

def get_id_mappings(conn):
    """Get mappings of names to IDs for all label tables"""
    cursor = conn.cursor()
    
    # Get issue mappings
    cursor.execute("SELECT issue_id, issue_name FROM issues")
    issue_mapping = {row[1]: row[0] for row in cursor.fetchall()}
    
    # Get response type mappings
    cursor.execute("SELECT response_type_id, response_type_name FROM response_types")
    response_type_mapping = {row[1]: row[0] for row in cursor.fetchall()}
    
    # Get therapeutic approach mappings
    cursor.execute("SELECT approach_id, approach_name FROM therapeutic_approaches")
    approach_mapping = {row[1]: row[0] for row in cursor.fetchall()}
    
    return issue_mapping, response_type_mapping, approach_mapping

def populate_conversations_and_relationships(conn, labeled_data, issue_mapping, response_type_mapping, approach_mapping):
    """Populate the conversations table and junction tables"""
    cursor = conn.cursor()
    search_index_data = []
    
    # Insert conversations
    print(f"Inserting {len(labeled_data)} conversations and their relationships...")
    
    for idx, row in labeled_data.iterrows():
        # Insert into conversations table
        cursor.execute(
            "INSERT INTO conversations (conversation_id, context, response) VALUES (?, ?, ?)",
            (row['ConversationID'], row['Context'], row['Response'])
        )
        
        # Parse JSON label arrays
        issues = json.loads(row['issues'])
        response_types = json.loads(row['response_types'])
        approaches = json.loads(row['therapeutic_approaches'])
        
        # Insert issue relationships
        for issue in issues:
            if issue in issue_mapping:
                cursor.execute(
                    "INSERT INTO conversation_issues (conversation_id, issue_id) VALUES (?, ?)",
                    (row['ConversationID'], issue_mapping[issue])
                )
        
        # Insert response type relationships
        for response_type in response_types:
            if response_type in response_type_mapping:
                cursor.execute(
                    "INSERT INTO conversation_response_types (conversation_id, response_type_id) VALUES (?, ?)",
                    (row['ConversationID'], response_type_mapping[response_type])
                )
        
        # Insert therapeutic approach relationships
        for approach in approaches:
            if approach in approach_mapping:
                cursor.execute(
                    "INSERT INTO conversation_approaches (conversation_id, approach_id) VALUES (?, ?)",
                    (row['ConversationID'], approach_mapping[approach])
                )
        
        # Add to search index data (combine context and response for full-text search)
        search_text = f"{row['Context']} {row['Response']}"
        search_index_data.append((row['ConversationID'], search_text))
        
        # Commit in batches to prevent transaction overflow
        if idx % 100 == 0:
            conn.commit()
    
    # Commit remaining changes
    conn.commit()
    
    # Populate search index
    print("Populating search index...")
    cursor.executemany(
        "INSERT INTO search_index (conversation_id, text) VALUES (?, ?)",
        search_index_data
    )
    
    # Commit changes
    conn.commit()
    
    print("Conversations and relationships populated successfully.")
    return True

def create_search_views(conn):
    """Create views to simplify common search patterns"""
    cursor = conn.cursor()
    
    # Create a view for conversations with their issues
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS conversation_with_issues AS
    SELECT c.conversation_id, c.context, c.response, 
           GROUP_CONCAT(i.issue_name, ', ') as issues
    FROM conversations c
    LEFT JOIN conversation_issues ci ON c.conversation_id = ci.conversation_id
    LEFT JOIN issues i ON ci.issue_id = i.issue_id
    GROUP BY c.conversation_id
    ''')
    
    # Create a view for conversations with their response types
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS conversation_with_response_types AS
    SELECT c.conversation_id, c.context, c.response, 
           GROUP_CONCAT(rt.response_type_name, ', ') as response_types
    FROM conversations c
    LEFT JOIN conversation_response_types crt ON c.conversation_id = crt.conversation_id
    LEFT JOIN response_types rt ON crt.response_type_id = rt.response_type_id
    GROUP BY c.conversation_id
    ''')
    
    # Create a view for conversations with their therapeutic approaches
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS conversation_with_approaches AS
    SELECT c.conversation_id, c.context, c.response, 
           GROUP_CONCAT(a.approach_name, ', ') as approaches
    FROM conversations c
    LEFT JOIN conversation_approaches ca ON c.conversation_id = ca.conversation_id
    LEFT JOIN therapeutic_approaches a ON ca.approach_id = a.approach_id
    GROUP BY c.conversation_id
    ''')
    
    # Create a comprehensive view with all labels
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS conversation_with_all_labels AS
    SELECT c.conversation_id, c.context, c.response, 
           cwi.issues, cwrt.response_types, cwa.approaches
    FROM conversations c
    LEFT JOIN conversation_with_issues cwi ON c.conversation_id = cwi.conversation_id
    LEFT JOIN conversation_with_response_types cwrt ON c.conversation_id = cwrt.conversation_id
    LEFT JOIN conversation_with_approaches cwa ON c.conversation_id = cwa.conversation_id
    ''')
    
    # Commit changes
    conn.commit()
    
    print("Search views created successfully.")
    return True

def verify_database(db_path):
    """Verify the database was populated correctly"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\nDatabase Verification:")
    
    # Check conversations count
    cursor.execute("SELECT COUNT(*) FROM conversations")
    print(f"Total conversations: {cursor.fetchone()[0]}")
    
    # Check issues count
    cursor.execute("SELECT COUNT(*) FROM issues")
    print(f"Unique issues: {cursor.fetchone()[0]}")
    
    # Check response types count
    cursor.execute("SELECT COUNT(*) FROM response_types")
    print(f"Unique response types: {cursor.fetchone()[0]}")
    
    # Check therapeutic approaches count
    cursor.execute("SELECT COUNT(*) FROM therapeutic_approaches")
    print(f"Unique therapeutic approaches: {cursor.fetchone()[0]}")
    
    # Test a simple search query
    cursor.execute('''
    SELECT conversation_id, context, response 
    FROM conversations 
    LIMIT 5
    ''')
    sample_conversations = cursor.fetchall()
    print(f"\nSample conversations: {len(sample_conversations)}")
    
    # Test a search index query
    cursor.execute('''
    SELECT conversation_id
    FROM search_index 
    WHERE text MATCH 'anxiety'
    LIMIT 5
    ''')
    sample_search = cursor.fetchall()
    print(f"Sample 'anxiety' search results: {len(sample_search)}")
    
    # Close connection
    conn.close()
    
    return True

def main():
    # Set file paths
    input_file = "data/mental_health_conversations_labeled.csv"
    db_path = "data/mental_health_db.sqlite"
    
    # Load labeled data
    print(f"Loading labeled data from {input_file}...")
    labeled_data = pd.read_csv(input_file)
    
    # Create database schema
    create_database_schema(db_path)
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Populate label tables
    populate_label_tables(conn, labeled_data)
    
    # Get ID mappings
    issue_mapping, response_type_mapping, approach_mapping = get_id_mappings(conn)
    
    # Populate conversations and relationships
    populate_conversations_and_relationships(
        conn, labeled_data, issue_mapping, response_type_mapping, approach_mapping
    )
    
    # Create search views
    create_search_views(conn)
    
    # Close connection
    conn.close()
    
    # Verify database
    verify_database(db_path)
    
    print("\nDatabase setup completed successfully!")

if __name__ == "__main__":
    main()