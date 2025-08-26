#!/usr/bin/env python3
"""
Airtable Setup Script for Media AI Pipeline
Creates the required tables and fields structure
"""
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

BASE_ID = os.getenv('MEDIA_AIRTABLE_BASE_ID')
API_KEY = os.getenv('AIRTABLE_API_KEY')

if not BASE_ID or not API_KEY:
    print("❌ Missing Airtable credentials in .env file")
    exit(1)

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

def create_table(table_name, fields):
    """Create a table with specified fields"""
    url = f'https://api.airtable.com/v0/meta/bases/{BASE_ID}/tables'
    
    payload = {
        "name": table_name,
        "fields": fields
    }
    
    print(f"Creating table: {table_name}")
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Successfully created '{table_name}' table")
        return data['id']
    elif "DUPLICATE_TABLE_NAME" in response.text:
        print(f"⚠️  Table '{table_name}' already exists - skipping")
        return "existing"
    else:
        print(f"❌ Failed to create '{table_name}' table:")
        print(f"Status: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def setup_screenshots_table():
    """Set up the Screenshots table"""
    fields = [
        {"name": "Filename", "type": "singleLineText"},
        {"name": "File Path", "type": "singleLineText"},
        {"name": "File Size MB", "type": "number", "options": {"precision": 0}},
        {"name": "Created At", "type": "singleLineText"},
        {"name": "Modified At", "type": "singleLineText"},
        {"name": "Analyzed At", "type": "singleLineText"},
        {"name": "Last Updated", "type": "singleLineText"},
        {"name": "Content Type", "type": "singleLineText"},
        {"name": "Primary Application", "type": "singleLineText"},
        {"name": "Summary", "type": "multilineText"},
        {"name": "Extracted Text", "type": "multilineText"},
        {"name": "Code Detected", "type": "checkbox", "options": {"icon": "check", "color": "greenBright"}},
        {"name": "Programming Language", "type": "singleLineText"},
        {"name": "Error Detected", "type": "checkbox", "options": {"icon": "check", "color": "redBright"}},
        {"name": "Error Details", "type": "multilineText"},
        {"name": "Technical Details", "type": "multilineText"},
        {"name": "Workflow Step", "type": "singleLineText"},
        {"name": "UI Elements", "type": "multilineText"},
        {"name": "Context Tags", "type": "multilineText"},
        {"name": "Searchable Keywords", "type": "multilineText"},
        {"name": "Full Analysis JSON", "type": "multilineText"}
    ]
    return create_table("Screenshots", fields)

def setup_recordings_table():
    """Set up the Screen Recordings table"""
    fields = [
        {"name": "Filename", "type": "singleLineText"},
        {"name": "File Path", "type": "singleLineText"},
        {"name": "File Size MB", "type": "number", "options": {"precision": 0}},
        {"name": "Duration Seconds", "type": "number", "options": {"precision": 0}},
        {"name": "Duration Formatted", "type": "singleLineText"},
        {"name": "Width", "type": "number", "options": {"precision": 0}},
        {"name": "Height", "type": "number", "options": {"precision": 0}},
        {"name": "FPS", "type": "number", "options": {"precision": 0}},
        {"name": "Codec", "type": "singleLineText"},
        {"name": "Bitrate", "type": "number", "options": {"precision": 0}},
        {"name": "Created At", "type": "singleLineText"},
        {"name": "Modified At", "type": "singleLineText"},
        {"name": "Analyzed At", "type": "singleLineText"},
        {"name": "Last Updated", "type": "singleLineText"},
        {"name": "Content Type", "type": "singleLineText"},
        {"name": "Primary Application", "type": "singleLineText"},
        {"name": "Summary", "type": "multilineText"},
        {"name": "Code Detected", "type": "checkbox", "options": {"icon": "check", "color": "greenBright"}},
        {"name": "Errors Detected", "type": "checkbox", "options": {"icon": "check", "color": "redBright"}},
        {"name": "Error Count", "type": "number", "options": {"precision": 0}},
        {"name": "Keyframe Count", "type": "number", "options": {"precision": 0}},
        {"name": "Context Tags", "type": "multilineText"},
        {"name": "Technical Details", "type": "multilineText"},
        {"name": "Workflow Steps", "type": "multilineText"},
        {"name": "Searchable Keywords", "type": "multilineText"},
        {"name": "Full Analysis JSON", "type": "multilineText"},
        {"name": "Keyframes JSON", "type": "multilineText"},
        {"name": "Total Tokens Used", "type": "number", "options": {"precision": 0}}
    ]
    return create_table("Screen Recordings", fields)

def setup_analysis_table():
    """Set up the Media Analysis table"""
    fields = [
        {"name": "Analysis ID", "type": "singleLineText"},
        {"name": "Media Type", "type": "singleLineText"},
        {"name": "Analysis Date", "type": "singleLineText"},
        {"name": "Status", "type": "singleLineText"}
    ]
    return create_table("Media Analysis", fields)

def setup_tags_table():
    """Set up the Tags table"""
    fields = [
        {"name": "Tag Name", "type": "singleLineText"},
        {"name": "Category", "type": "singleLineText"},
        {"name": "Usage Count", "type": "number", "options": {"precision": 0}}
    ]
    return create_table("Tags", fields)

def main():
    print("🚀 Setting up Airtable tables for Media AI Pipeline...")
    print(f"Base ID: {BASE_ID}")
    print()
    
    # Create all required tables
    tables_created = []
    
    # Screenshots table
    if setup_screenshots_table():
        tables_created.append("Screenshots")
    
    # Screen Recordings table  
    if setup_recordings_table():
        tables_created.append("Screen Recordings")
    
    # Media Analysis table
    if setup_analysis_table():
        tables_created.append("Media Analysis")
    
    # Tags table
    if setup_tags_table():
        tables_created.append("Tags")
    
    print()
    print("📊 Setup Summary:")
    print(f"✅ Successfully created {len(tables_created)} tables:")
    for table in tables_created:
        print(f"   - {table}")
    
    if len(tables_created) == 4:
        print()
        print("🎉 Airtable setup complete! You can now:")
        print("   1. Run 'python media_pipeline.py status' to verify connection")
        print("   2. Start analyzing media files")
        print("   3. View results in your Airtable base")
    else:
        print()
        print("⚠️  Some tables failed to create. Check the errors above.")

if __name__ == "__main__":
    main()
