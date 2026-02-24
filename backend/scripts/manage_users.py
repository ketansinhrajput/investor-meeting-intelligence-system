"""
CLI tool for managing users in the Call Transcript Intelligence System.

Usage:
  python scripts/manage_users.py create <username> <password> [--role admin|user]
  python scripts/manage_users.py list
  python scripts/manage_users.py delete <username>
  python scripts/manage_users.py change-password <username> <new_password>
  python scripts/manage_users.py activate <username>
  python scripts/manage_users.py deactivate <username>

Run without arguments for interactive mode:
  python scripts/manage_users.py

Run from the backend/ directory:
  cd backend
  python scripts/manage_users.py create admin secretpass --role admin
"""

import sys
import os

# Add backend directory to path so imports work
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BACKEND_DIR)

# Load .env from backend directory before any app imports
from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND_DIR, ".env"))

from datetime import datetime

from database import SessionLocal, init_db
from models.user import User
from api.auth import hash_password


# =============================================================================
# Core functions (used by both CLI and interactive mode)
# =============================================================================

def create_user(username: str, password: str, role: str = "user"):
    init_db()
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            print(f"Error: User '{username}' already exists.")
            sys.exit(1)

        user = User(
            username=username,
            password_hash=hash_password(password),
            role=role,
            is_active=True,
            created_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()
        print(f"Created user '{username}' with role '{role}'.")
    finally:
        db.close()


def list_users():
    init_db()
    db = SessionLocal()
    try:
        users = db.query(User).all()
        if not users:
            print("No users found.")
            return

        print(f"{'ID':<5} {'Username':<20} {'Role':<10} {'Active':<8} {'Created':<20} {'Last Login':<20}")
        print("-" * 83)
        for u in users:
            created = u.created_at.strftime("%Y-%m-%d %H:%M") if u.created_at else "-"
            last_login = u.last_login.strftime("%Y-%m-%d %H:%M") if u.last_login else "Never"
            print(f"{u.id:<5} {u.username:<20} {u.role:<10} {'Yes' if u.is_active else 'No':<8} {created:<20} {last_login:<20}")
    finally:
        db.close()


def delete_user(username: str):
    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            print(f"Error: User '{username}' not found.")
            sys.exit(1)

        db.delete(user)
        db.commit()
        print(f"Deleted user '{username}'.")
    finally:
        db.close()


def change_password(username: str, new_password: str):
    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            print(f"Error: User '{username}' not found.")
            sys.exit(1)

        user.password_hash = hash_password(new_password)
        db.commit()
        print(f"Password changed for user '{username}'.")
    finally:
        db.close()


def set_active(username: str, active: bool):
    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            print(f"Error: User '{username}' not found.")
            sys.exit(1)

        user.is_active = active
        db.commit()
        status = "activated" if active else "deactivated"
        print(f"User '{username}' {status}.")
    finally:
        db.close()


# =============================================================================
# Interactive mode helpers
# =============================================================================

def _is_last_active_admin(username: str) -> bool:
    """Check if this user is the only active admin."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user or user.role != "admin" or not user.is_active:
            return False
        active_admin_count = db.query(User).filter(
            User.role == "admin", User.is_active == True
        ).count()
        return active_admin_count <= 1
    finally:
        db.close()


def _user_exists(username: str) -> bool:
    db = SessionLocal()
    try:
        return db.query(User).filter(User.username == username).first() is not None
    finally:
        db.close()


def _prompt_username(action_label: str) -> str:
    """Prompt for a non-empty username. Returns the username or empty string to cancel."""
    while True:
        username = input(f"  Username to {action_label}: ").strip()
        if not username:
            print("  Username cannot be empty.")
            continue
        return username


def _confirm(prompt: str) -> bool:
    """Ask yes/no confirmation. Returns True for yes."""
    while True:
        answer = input(f"  {prompt} [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no", ""):
            return False


# =============================================================================
# Interactive mode functions
# =============================================================================

def create_user_interactive():
    print("\n  --- Create User ---")
    while True:
        username = input("  Username: ").strip()
        if not username:
            print("  Username cannot be empty.")
            continue
        break

    if _user_exists(username):
        print(f"  Error: User '{username}' already exists.")
        return

    while True:
        password = input("  Password: ").strip()
        if not password:
            print("  Password cannot be empty.")
            continue
        confirm = input("  Confirm password: ").strip()
        if password != confirm:
            print("  Passwords do not match. Try again.")
            continue
        break

    while True:
        role = input("  Role [admin/user] (default=user): ").strip().lower() or "user"
        if role in ("admin", "user"):
            break
        print("  Invalid role. Must be 'admin' or 'user'.")

    create_user(username, password, role)


def delete_user_interactive():
    print("\n  --- Delete User ---")
    list_users()
    print()
    username = _prompt_username("delete")

    if not _user_exists(username):
        print(f"  Error: User '{username}' not found.")
        return

    if _is_last_active_admin(username):
        print(f"  Error: Cannot delete '{username}' — they are the last active admin.")
        return

    if not _confirm(f"Are you sure you want to delete user '{username}'? This cannot be undone."):
        print("  Cancelled.")
        return

    delete_user(username)


def change_password_interactive():
    print("\n  --- Change Password ---")
    list_users()
    print()
    username = _prompt_username("change password for")

    if not _user_exists(username):
        print(f"  Error: User '{username}' not found.")
        return

    while True:
        password = input("  New password: ").strip()
        if not password:
            print("  Password cannot be empty.")
            continue
        confirm = input("  Confirm new password: ").strip()
        if password != confirm:
            print("  Passwords do not match. Try again.")
            continue
        break

    change_password(username, password)


def activate_user_interactive():
    print("\n  --- Activate User ---")
    list_users()
    print()
    username = _prompt_username("activate")

    if not _user_exists(username):
        print(f"  Error: User '{username}' not found.")
        return

    set_active(username, True)


def deactivate_user_interactive():
    print("\n  --- Deactivate User ---")
    list_users()
    print()
    username = _prompt_username("deactivate")

    if not _user_exists(username):
        print(f"  Error: User '{username}' not found.")
        return

    if _is_last_active_admin(username):
        print(f"  Error: Cannot deactivate '{username}' — they are the last active admin.")
        return

    if not _confirm(f"Are you sure you want to deactivate user '{username}'?"):
        print("  Cancelled.")
        return

    set_active(username, False)


# =============================================================================
# Interactive menu
# =============================================================================

MENU_OPTIONS = {
    "1": ("Create user", create_user_interactive),
    "2": ("List users", list_users),
    "3": ("Delete user", delete_user_interactive),
    "4": ("Change password", change_password_interactive),
    "5": ("Activate user", activate_user_interactive),
    "6": ("Deactivate user", deactivate_user_interactive),
}


def launch_interactive_menu():
    init_db()
    print()
    print("=" * 50)
    print("  Call Transcript Intelligence")
    print("  User Management")
    print("=" * 50)

    while True:
        print()
        for key, (label, _) in MENU_OPTIONS.items():
            print(f"  {key}) {label}")
        print(f"  0) Exit")
        print()

        try:
            choice = input("  Select an option: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Exiting.")
            break

        if choice == "0":
            print("\n  Goodbye.")
            break

        if choice in MENU_OPTIONS:
            _, handler = MENU_OPTIONS[choice]
            try:
                print()
                handler()
            except (KeyboardInterrupt, EOFError):
                print("\n  Operation cancelled.")
            except SystemExit:
                # Core functions call sys.exit(1) on errors — catch it
                # so the interactive loop continues.
                pass
        else:
            print("  Invalid option. Please try again.")


# =============================================================================
# CLI usage
# =============================================================================

def print_usage():
    print("Usage:")
    print("  python scripts/manage_users.py                                     (interactive mode)")
    print("  python scripts/manage_users.py create <username> <password> [--role admin|user]")
    print("  python scripts/manage_users.py list")
    print("  python scripts/manage_users.py delete <username>")
    print("  python scripts/manage_users.py change-password <username> <new_password>")
    print("  python scripts/manage_users.py activate <username>")
    print("  python scripts/manage_users.py deactivate <username>")


if __name__ == "__main__":
    args = sys.argv[1:]

    # No arguments → interactive mode
    if not args:
        launch_interactive_menu()
        sys.exit(0)

    command = args[0]

    if command == "create":
        if len(args) < 3:
            print("Error: create requires <username> <password>")
            sys.exit(1)
        username = args[1]
        password = args[2]
        role = "user"
        if "--role" in args:
            role_idx = args.index("--role")
            if role_idx + 1 < len(args):
                role = args[role_idx + 1]
                if role not in ("admin", "user"):
                    print("Error: role must be 'admin' or 'user'")
                    sys.exit(1)
        create_user(username, password, role)

    elif command == "list":
        list_users()

    elif command == "delete":
        if len(args) < 2:
            print("Error: delete requires <username>")
            sys.exit(1)
        delete_user(args[1])

    elif command == "change-password":
        if len(args) < 3:
            print("Error: change-password requires <username> <new_password>")
            sys.exit(1)
        change_password(args[1], args[2])

    elif command == "activate":
        if len(args) < 2:
            print("Error: activate requires <username>")
            sys.exit(1)
        set_active(args[1], True)

    elif command == "deactivate":
        if len(args) < 2:
            print("Error: deactivate requires <username>")
            sys.exit(1)
        set_active(args[1], False)

    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)
