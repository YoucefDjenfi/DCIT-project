# database/connection.py  — lazy init, sync get_connection
# Place this file at database/connection.py (overwrite the existing one)

from psycopg2 import pool
from config import Config
from dotenv import load_dotenv

load_dotenv()


class Database:
    def __init__(self):
        self.connection_pool = None

    def _ensure_pool(self) -> bool:
        if self.connection_pool is not None:
            return True
        if not Config.DATABASE_URL and not Config.DB_HOST:
            return False
        try:
            self.connection_pool = pool.ThreadedConnectionPool(
                minconn=Config.DB_POOL_MIN,
                maxconn=Config.DB_POOL_MAX,
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                database=Config.DB_NAME,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
            )
            print("Connection pool created successfully")
            return True
        except Exception as e:
            print(f"Connection pool failed: {e}")
            return False

    def get_connection(self):
        """Sync method — returns None if no DB configured."""
        if not self._ensure_pool():
            return None
        return self.connection_pool.getconn()

    def return_connection(self, connection):
        if connection and self.connection_pool:
            self.connection_pool.putconn(connection)

    def close_all_connections(self):
        if self.connection_pool:
            self.connection_pool.closeall()
            print("All connections closed")


db = Database()
