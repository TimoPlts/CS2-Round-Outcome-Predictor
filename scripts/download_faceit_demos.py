from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
import argparse
import gzip
import json
import os
import re
import shutil
import sys
from typing import Any
from urllib import error, parse, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cs2_round_predictor.config import RAW_DEMOS_DIR


DATA_API_BASE_URL = "https://open.faceit.com/data/v4"
DOWNLOADS_API_BASE_URL = "https://open.faceit.com"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "faceit_demo_manifest.json"
DEFAULT_GAME_ID = "cs2"
DEFAULT_PAGE_SIZE = 100
KNOWN_MAPS = {
    "ancient",
    "anubis",
    "cache",
    "cobblestone",
    "dust2",
    "inferno",
    "mirage",
    "nuke",
    "overpass",
    "train",
    "tuscan",
    "vertigo",
}


@dataclass(slots=True)
class DemoCandidate:
    nickname: str
    player_id: str
    match_id: str
    map_name: str | None
    finished_at: str | None
    demo_resource_urls: list[str]


class FaceitApiClient:
    def __init__(
        self,
        *,
        data_api_key: str,
        downloads_token: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self._data_api_key = data_api_key
        self._downloads_token = downloads_token
        self._timeout_seconds = timeout_seconds

    def get_player(self, nickname: str, *, game: str) -> dict[str, Any]:
        query = parse.urlencode({"nickname": nickname, "game": game})
        return self._request_json(
            f"{DATA_API_BASE_URL}/players?{query}",
            bearer_token=self._data_api_key,
        )

    def get_player_history(
        self,
        player_id: str,
        *,
        game: str,
        limit: int,
        from_timestamp: int | None,
        to_timestamp: int | None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        offset = 0

        while len(items) < limit:
            page_limit = min(DEFAULT_PAGE_SIZE, limit - len(items))
            query_params: dict[str, Any] = {
                "game": game,
                "limit": page_limit,
                "offset": offset,
            }
            if from_timestamp is not None:
                query_params["from"] = from_timestamp
            if to_timestamp is not None:
                query_params["to"] = to_timestamp

            query = parse.urlencode(query_params)
            payload = self._request_json(
                f"{DATA_API_BASE_URL}/players/{player_id}/history?{query}",
                bearer_token=self._data_api_key,
            )
            page_items = payload.get("items", [])
            if not isinstance(page_items, list) or not page_items:
                break

            items.extend(page_items)
            if len(page_items) < page_limit:
                break
            offset += len(page_items)

        return items[:limit]

    def get_match_details(self, match_id: str) -> dict[str, Any]:
        return self._request_json(
            f"{DATA_API_BASE_URL}/matches/{match_id}",
            bearer_token=self._data_api_key,
        )

    def get_signed_demo_download_url(self, resource_url: str) -> str:
        if not self._downloads_token:
            raise RuntimeError(
                "FACEIT Downloads API token is missing. Set FACEIT_DOWNLOADS_API_TOKEN first."
            )

        payload = self._request_json(
            f"{DOWNLOADS_API_BASE_URL}/download/v2/demos/download",
            bearer_token=self._downloads_token,
            method="POST",
            json_body={"resource_url": resource_url},
        )
        try:
            return str(payload["payload"]["download_url"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "FACEIT Downloads API response did not contain payload.download_url."
            ) from exc

    def download_file(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=self._timeout_seconds) as response:
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)

    def _request_json(
        self,
        url: str,
        *,
        bearer_token: str,
        method: str = "GET",
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Accept": "application/json",
        }
        data: bytes | None = None
        if json_body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(json_body).encode("utf-8")

        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self._timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"FACEIT request failed with {exc.code} for {url}: {body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"FACEIT request failed for {url}: {exc.reason}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download FACEIT CS2 demos for one or more players, filtered by a specific map."
        )
    )
    parser.add_argument(
        "--players",
        nargs="+",
        required=True,
        help="One or more FACEIT nicknames.",
    )
    parser.add_argument(
        "--map",
        required=True,
        dest="map_name",
        help="Map filter, for example inferno or de_inferno.",
    )
    parser.add_argument(
        "--game",
        default=DEFAULT_GAME_ID,
        help="FACEIT game id. Defaults to cs2.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=100,
        help="Maximum history matches to inspect per player.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="How many recent days of history to inspect.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DEMOS_DIR,
        help="Directory where demo files should be stored.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Where to write the JSON manifest of selected FACEIT matches.",
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="Optional cap on the number of selected matches to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite files even if they already exist.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only build the manifest, do not download demo files.",
    )
    return parser


def _normalize_map_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    normalized = normalized.removeprefix("de_")
    normalized = normalized.replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "dust": "dust2",
        "dustii": "dust2",
        "dustiii": "dust2",
        "dustii2": "dust2",
        "dustii_2": "dust2",
        "dustiiii": "dust2",
        "dust2": "dust2",
    }
    return aliases.get(normalized, normalized)


def _history_finished_at(item: dict[str, Any]) -> str | None:
    stats = item.get("stats")
    if isinstance(stats, dict):
        for key in ["Match Finished At", "Finished At", "finished_at"]:
            value = stats.get(key)
            if isinstance(value, str) and value.strip():
                return value
    for key in ["finished_at", "started_at", "created_at"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_match_id(item: dict[str, Any]) -> str | None:
    for key in ["match_id", "matchId"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    stats = item.get("stats")
    if isinstance(stats, dict):
        for key in ["Match Id", "match_id", "matchId"]:
            value = stats.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _extract_history_map(item: dict[str, Any]) -> str | None:
    stats = item.get("stats")
    if isinstance(stats, dict):
        for key in ["Map", "map", "Map Name", "map_name"]:
            value = stats.get(key)
            normalized = _normalize_map_name(value if isinstance(value, str) else None)
            if normalized:
                return normalized
    return None


def _collect_map_candidates(value: Any, *, parent_key: str | None = None) -> list[str]:
    candidates: list[str] = []
    if isinstance(value, dict):
        for key, nested_value in value.items():
            candidates.extend(_collect_map_candidates(nested_value, parent_key=key))
        return candidates
    if isinstance(value, list):
        for nested_value in value:
            candidates.extend(_collect_map_candidates(nested_value, parent_key=parent_key))
        return candidates
    if isinstance(value, str):
        if parent_key and "map" in parent_key.lower():
            normalized = _normalize_map_name(value)
            if normalized:
                candidates.append(normalized)
        else:
            for raw_match in re.findall(r"(de_[a-z0-9_]+)", value.lower()):
                normalized = _normalize_map_name(raw_match)
                if normalized:
                    candidates.append(normalized)
    return candidates


def _extract_match_map(match_details: dict[str, Any]) -> str | None:
    candidates = _collect_map_candidates(match_details)
    for candidate in candidates:
        if candidate in KNOWN_MAPS:
            return candidate
    return candidates[0] if candidates else None


def _extract_demo_urls(match_details: dict[str, Any]) -> list[str]:
    raw_demo_urls = match_details.get("demo_url")
    if isinstance(raw_demo_urls, str) and raw_demo_urls.strip():
        return [raw_demo_urls.strip()]
    if isinstance(raw_demo_urls, list):
        return [
            demo_url.strip()
            for demo_url in raw_demo_urls
            if isinstance(demo_url, str) and demo_url.strip()
        ]
    return []


def _sanitize_segment(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "unknown"


def _build_output_paths(output_dir: Path, match_id: str, resource_url: str) -> tuple[Path, Path | None]:
    resource_name = Path(parse.urlparse(resource_url).path).name or f"{match_id}.dem.gz"
    gz_path = output_dir / f"{_sanitize_segment(match_id)}_{_sanitize_segment(resource_name)}"
    dem_path = None
    if gz_path.suffix == ".gz":
        dem_path = gz_path.with_suffix("")
    return gz_path, dem_path


def _write_manifest(manifest_path: Path, demos: list[DemoCandidate]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(demo) for demo in demos]
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_history_window(days_back: int) -> tuple[int, int]:
    now = datetime.now(UTC)
    from_dt = now - timedelta(days=days_back)
    return int(from_dt.timestamp()), int(now.timestamp())


def _select_demo_candidates(
    client: FaceitApiClient,
    *,
    nicknames: list[str],
    game: str,
    map_name: str,
    history_limit: int,
    days_back: int,
) -> list[DemoCandidate]:
    normalized_target_map = _normalize_map_name(map_name)
    if normalized_target_map is None:
        raise ValueError("Map filter cannot be empty.")

    from_timestamp, to_timestamp = _resolve_history_window(days_back)
    selected_by_match: dict[str, DemoCandidate] = {}

    for nickname in nicknames:
        player = client.get_player(nickname, game=game)
        player_id = str(player["player_id"])
        history_items = client.get_player_history(
            player_id,
            game=game,
            limit=history_limit,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )

        for history_item in history_items:
            match_id = _extract_match_id(history_item)
            if not match_id or match_id in selected_by_match:
                continue

            history_map = _extract_history_map(history_item)
            if history_map and history_map != normalized_target_map:
                continue

            match_details = client.get_match_details(match_id)
            resolved_map = _extract_match_map(match_details) or history_map
            if resolved_map != normalized_target_map:
                continue

            demo_urls = _extract_demo_urls(match_details)
            if not demo_urls:
                continue

            selected_by_match[match_id] = DemoCandidate(
                nickname=nickname,
                player_id=player_id,
                match_id=match_id,
                map_name=resolved_map,
                finished_at=_history_finished_at(history_item),
                demo_resource_urls=demo_urls,
            )

    return list(selected_by_match.values())


def _download_demos(
    client: FaceitApiClient,
    *,
    demos: list[DemoCandidate],
    output_dir: Path,
    force: bool,
) -> tuple[int, int]:
    downloaded_files = 0
    skipped_files = 0

    for demo in demos:
        for resource_url in demo.demo_resource_urls:
            gz_path, dem_path = _build_output_paths(output_dir, demo.match_id, resource_url)

            if dem_path is not None and dem_path.exists() and not force:
                skipped_files += 1
                print(f"Skipping existing demo: {dem_path.name}")
                continue
            if gz_path.exists() and dem_path is None and not force:
                skipped_files += 1
                print(f"Skipping existing demo: {gz_path.name}")
                continue

            signed_url = client.get_signed_demo_download_url(resource_url)
            client.download_file(signed_url, gz_path)

            if dem_path is not None:
                with gzip.open(gz_path, "rb") as source:
                    with dem_path.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                gz_path.unlink()
                print(f"Downloaded {demo.match_id} -> {dem_path.name}")
            else:
                print(f"Downloaded {demo.match_id} -> {gz_path.name}")
            downloaded_files += 1

    return downloaded_files, skipped_files


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    data_api_key = os.environ.get("FACEIT_DATA_API_KEY")
    if not data_api_key:
        parser.error("FACEIT_DATA_API_KEY environment variable is required.")

    downloads_token = os.environ.get("FACEIT_DOWNLOADS_API_TOKEN")
    client = FaceitApiClient(
        data_api_key=data_api_key,
        downloads_token=downloads_token,
    )

    demos = _select_demo_candidates(
        client,
        nicknames=args.players,
        game=args.game,
        map_name=args.map_name,
        history_limit=args.history_limit,
        days_back=args.days_back,
    )

    if args.max_downloads is not None:
        demos = demos[: args.max_downloads]

    _write_manifest(args.manifest_path, demos)
    print(f"Selected matches: {len(demos)}")
    print(f"Saved manifest to {args.manifest_path}")

    if not demos:
        return 0

    if args.list_only:
        print("List-only mode enabled; no demos were downloaded.")
        return 0

    if not downloads_token:
        print(
            "FACEIT_DOWNLOADS_API_TOKEN is not set, so only the manifest was created. "
            "Set that token to download the actual demos."
        )
        return 0

    downloaded_files, skipped_files = _download_demos(
        client,
        demos=demos,
        output_dir=args.output_dir,
        force=args.force,
    )
    print(f"Downloaded files: {downloaded_files}")
    print(f"Skipped files: {skipped_files}")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
