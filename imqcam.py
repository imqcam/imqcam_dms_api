"""imqcam: thin helpers for pulling IMQCAM DMS form data into pandas.

The analysis lives in the notebooks. This module only covers the parts that
would otherwise be copy-pasted into every one of them: authentication,
paginated fetching, and turning the DMS's nested `data` payloads into frames.
"""

import os
import re

import pandas as pd
from dotenv import dotenv_values, find_dotenv, load_dotenv
from girder_client import GirderClient

DEFAULT_API_URL = "https://data.imqcam.org/api/v1"

# The nine forms registered on the DMS, keyed by formId. Entries carry a
# formId but not the form's name, so this is the lookup the notebooks use.
# Regenerate with `list_forms()` -- see tutorials/02_form_catalog.ipynb.
FORM_NAMES = {
    "663e6d21b18fa1c426e939ab": "Raw Powder Details",
    "66425a71b18fa1c426e93aa0": "Printer Build",
    "67d39472366ec49ab59dd4db": "NASA Ti64 30um Layers Data",
    "68922e35f5b193b7d3e07f5b": "Archival ULI Build (simplified)",
    "68922e94f5b193b7d3e07f5c": "Four-point flexural test",
    "68ed0eb251d2cf4a1aa1d14f": "Micromechanical Simulation Data",
    "6970da157f6ebb8fb320705c": "AM Build Parameters",
    "69fa18be514cf501621588cc": "Sample Heat Treatment Record",
    "69fa1d0872a32de5fe95028b": "Fractography Record",
}

# Envelope fields every entry carries, regardless of which form produced it.
ENVELOPE_FIELDS = ("_id", "uniqueId", "formId", "created", "updated", "folderId")

# IGSNs look like CMXMAL00007, optionally with a -NN specimen suffix.
IGSN_PATTERN = re.compile(r"\b([A-Z]{6}\d{5}(?:-\d+)?)\b")


def get_client(api_url=None):
    """Return an authenticated GirderClient for the IMQCAM DMS.

    Resolution order is `api_url` argument, then a repo-local `.env`, then
    DEFAULT_API_URL. An ambient GIRDER_API_URL is deliberately ignored: a value
    left over from a neighbouring DMS (data.htmdec.org uses the same Girder
    API) otherwise redirects every request to a host where this key is invalid.

    Args:
        api_url: str, explicit API URL, highest precedence.
    Returns:
        An authenticated GirderClient.
    """
    env_path = find_dotenv(usecwd=True)
    env_file = dotenv_values(env_path) if env_path else {}
    if env_path:
        load_dotenv(env_path, override=True)

    api_key = env_file.get("GIRDER_API_KEY") or os.environ.get("GIRDER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GIRDER_API_KEY is not set. Add it to a .env file in the repo root "
            "or export it before starting Jupyter."
        )
    url = api_url or env_file.get("GIRDER_API_URL") or DEFAULT_API_URL
    client = GirderClient(apiUrl=url)
    client.authenticate(apiKey=api_key)
    return client


def list_forms(client):
    """Return {formId: name} for every form the DMS exposes."""
    return {form["_id"]: form["name"] for form in client.get("form", parameters={"limit": 1000})}


def fetch_entries(client, form_id=None, page_size=100, sort="created", sortdir=1,
                  **params):
    """Fetch form entries, paging until the API runs out.

    `limit` is a page-size ceiling, not pagination -- a bare `limit=1000`
    silently truncates once the collection outgrows it. This pages with offset
    instead, so the result is always complete.

    Filtering by `form_id` is done server-side by the `formId` query parameter,
    so only the requested form crosses the wire. Paging is explicitly sorted:
    offset paging over an unordered result set can repeat or skip rows between
    requests.

    Args:
        client: authenticated GirderClient
        form_id: str, optional formId to restrict the query to
        page_size: int, entries per request
        sort: str, field to sort by
        sortdir: int, 1 ascending or -1 descending
        **params: further query parameters passed through to /entry, e.g.
            `query` (a regex) together with `field` (the field to match on).
    Returns:
        list of entry dicts
    """
    entries = []
    offset = 0
    while True:
        query = {"limit": page_size, "offset": offset, "sort": sort, "sortdir": sortdir}
        if form_id is not None:
            query["formId"] = form_id
        query.update(params)
        page = client.get("entry", parameters=query)
        entries.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return entries


def entries_to_frame(entries, sep="."):
    """Flatten entries to one row per entry.

    The form-specific payload lives under `data`; nested objects are flattened
    with dotted names. Envelope fields are kept alongside, and `formName` is
    resolved from FORM_NAMES where known.

    Args:
        entries: list of entry dicts from fetch_entries
        sep: str, separator for nested keys
    Returns:
        DataFrame, one row per entry.
    """
    rows = []
    for entry in entries:
        row = {field: entry.get(field) for field in ENVELOPE_FIELDS}
        row["formName"] = FORM_NAMES.get(entry.get("formId"))
        row["igsn"] = extract_igsn(entry)
        rows.append(row)
    meta = pd.DataFrame(rows)
    payload = pd.json_normalize([e.get("data", {}) for e in entries], sep=sep)
    # Envelope columns win on a name clash (e.g. AM Build repeats `uniqueId`).
    payload = payload.drop(columns=[c for c in payload.columns if c in meta.columns])
    return pd.concat([meta, payload], axis=1)


def explode_records(entries, path, sep="."):
    """Flatten a list-valued field to one row per nested record.

    Several forms hold repeated measurements in a list: `tests` on the flexural
    and fractography forms, `buildParameters` on AM Build, `composition` on Raw
    Powder. This returns one row per element, carrying the parent's identity
    down so the result can be joined back or grouped.

    Args:
        entries: list of entry dicts from fetch_entries
        path: str, the key under `data` holding the list
        sep: str, separator for nested keys
    Returns:
        DataFrame, one row per nested record. Empty if nothing has the field.
    """
    records, parents = [], []
    for entry in entries:
        data = entry.get("data") or {}
        items = data.get(path)
        if not isinstance(items, list):
            continue
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            records.append(item)
            parents.append(
                {
                    "_id": entry.get("_id"),
                    "uniqueId": entry.get("uniqueId"),
                    "formId": entry.get("formId"),
                    "igsn": extract_igsn(entry),
                    "recordIndex": position,
                }
            )
    if not records:
        return pd.DataFrame()
    frame = pd.json_normalize(records, sep=sep)
    parent_frame = pd.DataFrame(parents)
    frame = frame.drop(columns=[c for c in frame.columns if c in parent_frame.columns])
    return pd.concat([parent_frame, frame], axis=1)


def extract_igsn(entry):
    """Pull the sample IGSN out of an entry, or None if it has none.

    Forms record it four different ways: `assignedIGSN` (Printer Build),
    `sampleIGSN` (Fractography), a "IGSN - depositionId - buildId" string in
    `lookup` (ULI, flexural), or a list of IGSNs in `lookup` (AM Build, heat
    treatment). NASA TTT entries carry no IGSN at all.
    """
    data = entry.get("data")
    if not isinstance(data, dict):
        return None
    for field in ("assignedIGSN", "sampleIGSN"):
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    lookup = data.get("lookup")
    candidates = lookup if isinstance(lookup, list) else [lookup]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        match = IGSN_PATTERN.search(candidate)
        if match:
            return match.group(1)
    return None
