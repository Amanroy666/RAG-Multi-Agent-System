"""
Database connection manager with connection pooling
"""
from contextlib import contextmanager
from typing import Generator
import psycopg2
from psycopg2 import pool
from .logger import setup_logger

logger = setup_logger(__name__)

class DatabaseManager:
    """PostgreSQL database connection manager"""
    
    def __init__(self, config: dict):
        self.config = config
        self._pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config.get('min_connections', 1),
                maxconn=self.config.get('max_connections', 10),
                host=self.config['host'],
                port=self.config.get('port', 5432),
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self) -> Generator:
        """Context manager for database connections"""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            self._pool.putconn(conn)
    
    def execute_query(self, query: str, params: tuple = None):
        """Execute a query and return results"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    return cursor.fetchall()
                return None
    
    def close(self):
        """Close all connections in pool"""
        if self._pool:
            self._pool.closeall()
            logger.info("Database connection pool closed")
