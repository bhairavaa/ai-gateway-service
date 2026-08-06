"""Bootstrap CLI: mint a new API key directly via the repository layer.

No HTTP admin surface — whoever can run this already has direct DB/filesystem
access, so there's no separate auth-for-auth problem to solve. Prints the raw
key exactly once; only its SHA-256 hash is ever persisted.

Usage:
    python scripts/create_api_key.py "ci-bot" --scope chat:write
"""

import argparse
import asyncio
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth.api_key_backend import hash_api_key  # noqa: E402
from app.db.repositories.api_key_repository import SqliteApiKeyRepository  # noqa: E402
from app.db.session import AsyncSessionLocal  # noqa: E402


async def create_api_key(name: str, scopes: list[str] | None = None) -> str:
    raw_key = f"agw_{secrets.token_urlsafe(32)}"
    async with AsyncSessionLocal() as session:
        repo = SqliteApiKeyRepository(session)
        await repo.create(key_hash=hash_api_key(raw_key), name=name, scopes=scopes or [])
    return raw_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new AI Gateway API key")
    parser.add_argument("name", help="Human-readable name for this key (e.g. 'ci-bot', 'dashboard')")
    parser.add_argument("--scope", action="append", dest="scopes", default=[], help="Optional scope (repeatable)")
    args = parser.parse_args()

    raw_key = asyncio.run(create_api_key(args.name, args.scopes))
    print(f"Created API key '{args.name}':\n")
    print(raw_key)
    print("\nStore this now - it will not be shown again.")


if __name__ == "__main__":
    main()
