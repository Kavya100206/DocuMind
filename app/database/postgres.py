"""
PostgreSQL Database Configuration

This file sets up the database connection using SQLAlchemy.

What is SQLAlchemy?
-------------------
SQLAlchemy is an ORM (Object-Relational Mapping) library.
It lets us work with databases using Python classes instead of raw SQL.

What is an ORM?
---------------
ORM = Object-Relational Mapping
- Objects = Python classes (User, Document, etc.)
- Relational = Database tables
- Mapping = Converts between Python objects and database rows

Example:
    # Instead of SQL:
    SELECT * FROM users WHERE id = 1;
    
    # We write Python:
    user = session.query(User).filter(User.id == 1).first()
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config.settings import settings

# Create database engine
# The engine is the connection to the database
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using them
    echo=settings.DEBUG   # Log SQL queries in debug mode
)

# Create SessionLocal class
# A session is like a "workspace" for database operations
# Each request will get its own session
SessionLocal = sessionmaker(
    autocommit=False,  # Don't auto-commit changes
    autoflush=False,   # Don't auto-flush changes
    bind=engine        # Bind to our database engine
)

# Create Base class for models
# All our database models will inherit from this
Base = declarative_base()


def get_db():
    """
    Database dependency for FastAPI
    
    What is a dependency?
    ---------------------
    In FastAPI, dependencies are reusable pieces of code.
    This function provides a database session to our endpoints.
    
    How it works:
    -------------
    1. Creates a new database session
    2. Yields it to the endpoint (gives it temporarily)
    3. Closes the session when the request is done
    
    Usage in a controller:
        @router.get("/users")
        def get_users(db: Session = Depends(get_db)):
            users = db.query(User).all()
            return users
    """
    db = SessionLocal()
    try:
        yield db  # Give the session to the endpoint
    finally:
        db.close()  # Always close the session when done


def init_db():
    """
    Initialize the database
    
    What does this do?
    ------------------
    Creates all tables defined in our models.
    This is called when the application starts.
    
    Note: In production, we'd use Alembic for migrations instead.
    """
    Base.metadata.create_all(bind=engine)
