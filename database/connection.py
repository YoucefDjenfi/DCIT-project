from psycopg2 import pool
from config import Config
from dotenv import load_dotenv

load_dotenv()


class Database:
    def __init__(self):
        # Don't connect here — connect lazily on first use.
        # This lets the bot start even without a PostgreSQL server running.
        self.connection_pool = None

    def _ensure_pool(self) -> bool:
        """
        Create the connection pool on first use.
        Returns True if ready, False if DB is not configured.
        Never raises — DB-dependent cogs check the return value.
        """
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
        """
        Returns a connection from the pool, or None if no DB is configured.
        Callers must check for None before using the connection.
        """
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


# Module-level instance — no connection attempt until get_connection() is called
db = Database()
