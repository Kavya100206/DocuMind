"""
Test database connection
Run this to verify your NeonDB connection works
"""

from sqlalchemy import create_engine, text
from app.config.settings import settings

print("Testing database connection...")
print(f"Database URL: {settings.DATABASE_URL[:50]}...")  # Print first 50 chars only

try:
    # Create engine
    engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
    
    # Test connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        version = result.fetchone()
        print(f"✅ Connection successful!")
        print(f"PostgreSQL version: {version[0]}")
        
except Exception as e:
    print(f"❌ Connection failed!")
    print(f"Error: {e}")
    print(f"\nTroubleshooting:")
    print("1. Check your DATABASE_URL in .env file")
    print("2. Verify NeonDB credentials are correct")
    print("3. Ensure your IP is allowed in NeonDB settings")
