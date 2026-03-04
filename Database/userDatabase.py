import sqlite3
import hashlib
import secrets
from datetime import datetime
from Source.Models.user_model import UserRecord, ALL_ROLES

DB_NAME = "clearscanUsers.db"


def _hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(32)
    combined = (salt + password).encode("utf-8")
    hashed   = hashlib.sha256(combined).hexdigest()
    return hashed, salt


def verify_password(password, stored_hash):
    if "$" not in stored_hash:
        return False
    salt, expected = stored_hash.split("$", 1)
    computed, _    = _hash_password(password, salt)
    return computed == expected


def make_password_hash(password):
    hashed, salt = _hash_password(password)
    return f"{salt}${hashed}"


class UserDatabase:
    def __init__(self):
        self.conn = sqlite3.connect(DB_NAME)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()
        self._seed_default_admin()

    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                role          TEXT    NOT NULL,
                first_name    TEXT    NOT NULL,
                last_name     TEXT    NOT NULL,
                email         TEXT    DEFAULT '',
                is_active     INTEGER DEFAULT 1,
                created_at    TEXT    DEFAULT CURRENT_TIMESTAMP,
                last_login    TEXT    DEFAULT ''
            );
        """)
        self.conn.commit()

    def _seed_default_admin(self):
        count = self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            self.create_user(
                username   = "admin",
                password   = "admin123",
                role       = "sysadmin",
                first_name = "System",
                last_name  = "Administrator",
            )
            print("[UserDB] Default admin created -> username: admin  password: admin123")

    def create_user(self, username, password, role, first_name, last_name, email=""):
        if role not in ALL_ROLES:
            raise ValueError(f"Invalid role: {role}")
        pw_hash = make_password_hash(password)
        cursor  = self.conn.execute(
            "INSERT INTO users (username, password_hash, role, first_name, last_name, email) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (username.strip(), pw_hash, role, first_name.strip(), last_name.strip(), email.strip())
        )
        self.conn.commit()
        return self.fetch_by_id(cursor.lastrowid)

    def authenticate(self, username, password):
        row = self.conn.execute(
            "SELECT * FROM users WHERE username = ? AND is_active = 1",
            (username.strip(),)
        ).fetchone()
        if not row:
            return None
        if not verify_password(password, row["password_hash"]):
            return None
        self.conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row["id"])
        )
        self.conn.commit()
        return self._row_to_record(row)

    def fetch_by_id(self, user_id):
        row = self.conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def fetch_all_users(self):
        rows = self.conn.execute(
            "SELECT * FROM users ORDER BY last_name, first_name"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def username_exists(self, username):
        row = self.conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
        return row is not None

    def update_user(self, user):
        self.conn.execute(
            "UPDATE users SET username=?, first_name=?, last_name=?, email=?, role=?, is_active=? WHERE id=?",
            (user.username.strip(), user.first_name, user.last_name, user.email,
             user.role, int(user.is_active), user.id)
        )
        self.conn.commit()

    def change_password(self, user_id, new_password):
        self.conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (make_password_hash(new_password), user_id)
        )
        self.conn.commit()

    def deactivate_user(self, user_id):
        self.conn.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))
        self.conn.commit()

    @staticmethod
    def _row_to_record(row):
        return UserRecord(
            id            = row["id"],
            username      = row["username"],
            password_hash = row["password_hash"],
            role          = row["role"],
            first_name    = row["first_name"],
            last_name     = row["last_name"],
            email         = row["email"] or "",
            is_active     = bool(row["is_active"]),
            created_at    = row["created_at"] or "",
            last_login    = row["last_login"] or "",
        )