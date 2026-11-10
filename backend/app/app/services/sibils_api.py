"""What the SIBiLS fetch API returns, and what the code requires of its URL.

Standard library only, so the worker, the API, the readiness gate and the unit
tests share one reading of a `/fetch` response (shared triage guard, §1/§5):

- `sibils_base_url`: the configured base URL must be http(s); a trailing slash
  is stripped because the client appends `/fetch`;
- `parse_fetch_response`: one response body -> `FetchResult`: a record per id
  SIBiLS returned (`found`), the requested ids it did not (`missing`, from the
  set difference and its `warning: "missing ids:…"`), its warnings, and the
  instance and collection versions. "Not in the response" is a fact the caller
  records per article, never "no text, score 0".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


class SibilsFetchError(Exception):
    """The fetch failed (HTTP error, timeout, non-JSON body, success=false)."""


@dataclass
class FetchResult:
    found: Dict[str, Any] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sibils_version: Optional[str] = None
    collection_version: Optional[str] = None

    def merge(self, other: "FetchResult") -> None:
        self.found.update(other.found)
        self.missing.extend(i for i in other.missing if i not in self.missing)
        self.warnings.extend(w for w in other.warnings if w not in self.warnings)
        self.sibils_version = self.sibils_version or other.sibils_version
        self.collection_version = self.collection_version or other.collection_version

    @property
    def warning(self) -> Optional[str]:
        return "; ".join(self.warnings) or None


def sibils_base_url(raw: Optional[str]) -> str:
    """Validate and normalise SIBILS_URL: required, http(s), no trailing slash."""
    value = (raw or "").strip()
    if not value:
        raise RuntimeError(
            "SIBILS_URL is not set; refusing to start without a text source (no silent default)"
        )
    if not value.startswith(("http://", "https://")):
        raise RuntimeError(f"SIBILS_URL={value!r} must start with http:// or https://")
    return value.rstrip("/")


def parse_missing_ids(warning: Optional[str]) -> List[str]:
    """`"missing ids:1,2"` -> `["1", "2"]`; any other text -> `[]`."""
    m = re.search(r"missing ids?:\s*([\w,\s]+)", warning or "")
    if not m:
        return []
    return [i for i in re.split(r"[,\s]+", m.group(1)) if i]


def parse_fetch_response(data: Any, ids: Iterable[Any]) -> FetchResult:
    """One decoded `/fetch` body for `ids` -> FetchResult. Raises SibilsFetchError
    when the body is not a SIBiLS answer (wrong type, `success: false`)."""
    if not isinstance(data, dict):
        raise SibilsFetchError(f"unexpected body from SIBiLS: {type(data).__name__}")
    if data.get("success") is False:
        raise SibilsFetchError(f"SIBiLS error: {data.get('error') or 'success=false'}")

    result = FetchResult(
        sibils_version=data.get("sibils_version"),
        collection_version=data.get("collection_version"),
    )
    for item in data.get("sibils_article_set") or []:
        if isinstance(item, dict) and "_id" in item:
            result.found[str(item["_id"])] = item

    requested = [str(i) for i in ids]
    result.missing = [i for i in requested if i not in result.found]
    warning = str(data.get("warning") or "").strip()
    if warning:
        result.warnings.append(warning)
        for i in parse_missing_ids(warning):
            if i in requested and i not in result.missing:
                result.missing.append(i)
    return result
