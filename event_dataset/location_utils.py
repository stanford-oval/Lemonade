import time
from functools import lru_cache
from threading import Lock
from typing import Optional

import requests
from requests import JSONDecodeError

from event_dataset.cache import diskcache_cache
from log_utils import logger

OPEN_STREET_MAP_URL = "https://nominatim.openstreetmap.org"


rate_limit_lock = Lock()
last_request_time = 0


cc = None


@lru_cache
def country_name_to_code(country_name: str) -> str:
    global cc
    if not cc:
        import country_converter as coco

        cc = coco.CountryConverter()
    ret = cc.convert(names=country_name, to="ISO2")
    if isinstance(ret, list):
        return ret[0] if ret else ""
    return ret


@diskcache_cache
def search_location_in_openstreetmap(country_code: str, location_string: str):
    """
    Search for location information using OpenStreetMap's Nominatim API.
    """
    headers = {"User-Agent": "stanford-oval/Lemonade"}
    url = f"{OPEN_STREET_MAP_URL}/search?q={location_string}&format=json&addressdetails=1&limit=5&countrycodes={country_code}&accept-language=en"

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        try:
            data = response.json()
        except JSONDecodeError:
            logger.warning(f"Geocoding request failed for URL {url}")
            return None
        if not data:
            return None
        # Extract relevant information from the JSON response
        ret = data[0]["address"]
        ret = {
            k: v
            for k, v in ret.items()
            if not k.startswith("ISO") and k != "country_code"
        }

        return ret
    else:
        return None


@diskcache_cache
def search_coordinates_in_openstreetmap(
    longitude: float, latitude: float, zoom: int = 16, wait_seconds: float = 1
) -> Optional[dict]:
    """
    Search for location information using coordinates in OpenStreetMap using Nominatim's reverse API.
    See https://nominatim.org/release-docs/latest/api/Reverse/ for API documentation
    """
    _enforce_rate_limit(wait_seconds)

    headers = {"User-Agent": "stanford-oval/Lemonade"}
    url = f"{OPEN_STREET_MAP_URL}/reverse?lat={latitude}&lon={longitude}&zoom={zoom}&format=geojson&accept-language=en&addressdetails=1&layer=address"

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return None

    try:
        data = response.json()
    except JSONDecodeError:
        logger.warning(f"Geocoding request failed for URL {url}")
        return None

    if "features" not in data or not data["features"]:
        logger.warning(f"No features found for coordinates {latitude}, {longitude}")
        return None

    address = data["features"][0]["properties"]["address"]
    return {k: v for k, v in address.items() if not k.startswith("ISO")}


def _enforce_rate_limit(wait_seconds: float = 1) -> None:
    """Enforce rate limiting for API requests."""
    global last_request_time
    with rate_limit_lock:
        current_time = time.time()
        elapsed_time = current_time - last_request_time
        if elapsed_time < wait_seconds:
            time.sleep(wait_seconds - elapsed_time)
        last_request_time = time.time()
