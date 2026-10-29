"""Search and download helpers for IMQCAM file collections."""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileSearchSettings:
    igsns: tuple[str, ...]
    data_type: str
    composition: str
    file_extension: str
    page_size: int
    max_results: int | None


@dataclass(frozen=True)
class FileSearchResults:
    items: tuple[dict, ...]
    settings: FileSearchSettings


def split_igsns(value):
    """Normalize one or many IGSNs into a unique, ordered list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result = []
        for part in value:
            result.extend(split_igsns(part))
        return list(dict.fromkeys(result))

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return []
    return list(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))


def item_igsns(item):
    """Return all constituent IGSNs stored on one returned item."""
    return split_igsns((item.get("meta") or {}).get("igsn"))


def matches_requested_igsn(item, requested_igsns):
    """Return whether an item contains one of the requested IGSNs."""
    if not requested_igsns:
        return True
    stored = {value.casefold() for value in item_igsns(item)}
    requested = {value.casefold() for value in requested_igsns}
    return bool(stored & requested)


def matches_composition(item, composition):
    """Return whether item metadata matches the requested composition."""
    requested = str(composition or "").strip().casefold()
    if not requested:
        return True
    stored = str((item.get("meta") or {}).get("composition") or "").strip().casefold()
    return stored == requested


def normalize_extension(value):
    """Normalize ``ang`` or ``.ang`` to ``.ang``."""
    text = str(value or "").strip().lower()
    return text if not text or text.startswith(".") else f".{text}"


def matches_file_extension(item, extension):
    """Match an item's name suffix; folder contents are not inspected."""
    extension = normalize_extension(extension)
    if not extension:
        return True
    name = str(item.get("name") or "").strip().lower()
    return not name.startswith("._") and name.endswith(extension)


def _validate_nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def fetch_datafiles(
    client,
    igsn="",
    data_type="",
    composition="",
    file_extension="",
    page_size=100,
    max_results=None,
    progress_callback=None,
    progress_every=10,
):
    """Fetch matching items and preserve the normalized search settings."""
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
        raise ValueError("page_size must be a positive integer.")
    if max_results is not None:
        _validate_nonnegative_integer(max_results, "max_results")
    if isinstance(progress_every, bool) or not isinstance(progress_every, int) or progress_every <= 0:
        raise ValueError("progress_every must be a positive integer.")

    settings = FileSearchSettings(
        igsns=tuple(split_igsns(igsn)),
        data_type=str(data_type or "").strip(),
        composition=str(composition or "").strip(),
        file_extension=normalize_extension(file_extension),
        page_size=page_size,
        max_results=max_results,
    )
    if max_results == 0:
        return FileSearchResults(items=(), settings=settings)

    parameters = {
        "limit": page_size,
        "offset": 0,
        "extraFields": json.dumps(["meta.composition"]),
    }
    if settings.data_type:
        parameters["dataType"] = settings.data_type

    results = []
    seen_ids = set()
    seen_pages = set()
    offset = 0
    page_number = 0
    fetched = 0

    while True:
        parameters["offset"] = offset
        batch = client.get("imqcam/datafiles", parameters=parameters)
        if not isinstance(batch, list):
            raise TypeError(
                f"/imqcam/datafiles returned {type(batch).__name__}; expected a list."
            )
        if len(batch) > page_size:
            raise RuntimeError(
                f"/imqcam/datafiles returned {len(batch)} items for limit={page_size}; "
                "pagination cannot be trusted."
            )
        if not batch:
            break

        page_signature = tuple(
            str(item.get("_id"))
            if isinstance(item, dict) and item.get("_id") is not None
            else json.dumps(item, sort_keys=True, default=str)
            for item in batch
        )
        if page_signature in seen_pages:
            raise RuntimeError(
                f"Pagination repeated the page at offset {offset}; "
                "the endpoint may be ignoring offset."
            )
        seen_pages.add(page_signature)

        page_number += 1
        for item in batch:
            if not isinstance(item, dict):
                raise TypeError("Every /imqcam/datafiles result must be an object.")
            item_id = item.get("_id")
            if item_id is not None:
                if not isinstance(item_id, str):
                    raise TypeError("Each item _id must be a string when present.")
                if item_id in seen_ids:
                    raise RuntimeError(
                        f"Pagination returned duplicate item ID {item_id!r}; "
                        "results may have been skipped."
                    )
                seen_ids.add(item_id)

            if (
                matches_requested_igsn(item, settings.igsns)
                and matches_composition(item, settings.composition)
                and matches_file_extension(item, settings.file_extension)
            ):
                results.append(item)

        fetched += len(batch)
        reached_limit = (
            settings.max_results is not None
            and len(results) >= settings.max_results
        )
        final_page = len(batch) < page_size or reached_limit
        if progress_callback and (page_number % progress_every == 0 or final_page):
            progress_callback(page_number, fetched, len(results))

        if reached_limit:
            results = results[:settings.max_results]
            break
        if len(batch) < page_size:
            break
        offset += len(batch)

    return FileSearchResults(items=tuple(results), settings=settings)


def safe_folder_name(value, fallback):
    """Return one safe path component and reject dot-directory names."""
    text = str(value or "").strip() or fallback
    if text in {".", ".."}:
        raise ValueError("Folder names '.' and '..' are not allowed.")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    if safe in {".", ".."}:
        raise ValueError("Folder names '.' and '..' are not allowed.")
    return safe


def download_folder_for_results(root, settings):
    """Build the filter-specific destination from the settings used for the search."""
    if not isinstance(settings, FileSearchSettings):
        raise TypeError("settings must be the FileSearchSettings used for the search.")
    if not settings.igsns:
        igsn_folder = "all_igsns"
    elif len(settings.igsns) == 1:
        igsn_folder = settings.igsns[0]
    else:
        igsn_folder = "multiple_igsns"

    return (
        Path(root)
        / "downloads"
        / safe_folder_name(settings.data_type, "all_data_types")
        / safe_folder_name(igsn_folder, "all_igsns")
        / safe_folder_name(settings.composition, "all_compositions")
        / safe_folder_name(settings.file_extension, "all_extensions")
    )


def download_items(client, search_results, root, limit=5, progress_callback=print):
    """Download matched items into their shared filter-specific destination."""
    if not isinstance(search_results, FileSearchResults):
        raise TypeError("search_results must be returned by fetch_datafiles().")
    if limit is not None:
        _validate_nonnegative_integer(limit, "limit")

    items = search_results.items[:limit] if limit is not None else search_results.items
    destination = download_folder_for_results(root, search_results.settings)
    destination.mkdir(parents=True, exist_ok=True)
    if progress_callback:
        progress_callback(f"Downloading {len(items):,} matched item(s)...")
        progress_callback(f"Destination: {destination}")

    downloaded = []
    for number, item in enumerate(items, start=1):
        item_id = item.get("_id")
        item_name = item.get("name", item_id)
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"Item has an invalid or missing _id: {item_name!r}")
        client.downloadItem(item_id, dest=str(destination))
        downloaded.append(item_id)
        if progress_callback:
            progress_callback(f"[{number}/{len(items)}] {item_name}")

    if progress_callback:
        progress_callback("Download complete.")
    return tuple(downloaded)
