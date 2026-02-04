import os
import sqlite3

IMAGE_DB = "clearscanImage.db"
FINDINGS_DB = "clearscanFindings.db"


def reset_image_database():
    if not os.path.exists(IMAGE_DB):
        print(f"[SKIP] {IMAGE_DB} does not exist")
        return

    conn = sqlite3.connect(IMAGE_DB)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS images")

    cursor.execute("""
        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            user TEXT NOT NULL
        );
    """)

    conn.commit()
    conn.close()
    print(f"[OK] Reset {IMAGE_DB}")


def reset_findings_database():
    if not os.path.exists(FINDINGS_DB):
        print(f"[SKIP] {FINDINGS_DB} does not exist")
        return

    conn = sqlite3.connect(FINDINGS_DB)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS findings")

    cursor.execute("""
        CREATE TABLE findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            probability REAL NOT NULL,
            model_version TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()
    print(f"[OK] Reset {FINDINGS_DB}")


if __name__ == "__main__":
    print("⚠️  THIS WILL DELETE ALL DATABASE DATA ⚠️\n")

    reset_image_database()
    reset_findings_database()

    print("\n✅ All databases cleared successfully.")
