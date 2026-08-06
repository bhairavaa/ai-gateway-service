"""Auth abstraction.

ApiKeyAuthBackend is the only implementation today. Adding JWT/OAuth later
means a new AuthBackend implementation plus a config flag selecting which one
app/middleware/authentication.py constructs — no route/service changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class AuthContext:
    api_key_id: str
    name: str
    scopes: list[str]


class AuthenticationError(Exception):
    pass


class AuthBackend(ABC):
    @abstractmethod
    async def authenticate(self, credential: str) -> AuthContext:
        """Raise AuthenticationError on an invalid, inactive, expired, or unknown credential."""
