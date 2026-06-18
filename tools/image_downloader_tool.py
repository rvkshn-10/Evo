from langchain.tools import tool
import logging
import os
import json
import re
import requests

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
_TIMEOUT = 20
_MAX_IMAGES = 10
_ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


def _safe_filename(url: str, index: int, ext: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", url.split("/")[-1].split("?")[0])[:40]
    return f"image_{index:03d}_{slug or 'download'}{ext}"


@tool
def download_media_images(event_folder: str) -> str:
    """
    Read media_manifest.json from the event folder, download each image URL
    to {event_folder}/images/, and update each manifest entry with a
    'local_path' field pointing to the downloaded file.

    Args:
        event_folder: Path to the event output folder

    Returns:
        JSON array string of download results (url, local_path, title, status)
    """
    manifest_path = os.path.join(event_folder, "media_manifest.json")
    if not os.path.exists(manifest_path):
        return f"Error: media_manifest.json not found at {manifest_path}"

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as e:
        return f"Error reading media_manifest.json: {e}"

    images_dir = os.path.join(event_folder, "images")
    os.makedirs(images_dir, exist_ok=True)

    results = []
    downloaded = 0

    for i, entry in enumerate(entries[:_MAX_IMAGES]):
        url = entry.get("url", "")
        title = entry.get("title", f"Image {i + 1}")

        if not url or not url.startswith("http"):
            results.append({"url": url, "title": title, "status": "skipped — invalid URL", "local_path": None})
            continue

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT, stream=True)
            if not resp.ok:
                results.append({"url": url, "title": title, "status": f"HTTP {resp.status_code}", "local_path": None})
                continue

            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
            ext = _ALLOWED_TYPES.get(content_type)
            if not ext:
                # Try to infer from URL
                for mime, e in _ALLOWED_TYPES.items():
                    if any(url.lower().endswith(e) or e[1:] in url.lower() for e in [e]):
                        ext = e
                        break
                else:
                    ext = ".jpg"

            filename = _safe_filename(url, i + 1, ext)
            local_path = os.path.join(images_dir, filename)

            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            entry["local_path"] = os.path.relpath(local_path, event_folder).replace("\\", "/")
            results.append({
                "url": url,
                "title": title,
                "status": "downloaded",
                "local_path": entry["local_path"],
            })
            downloaded += 1
            logger.info(f"[download_media_images] Downloaded: {local_path}")

        except requests.exceptions.Timeout:
            results.append({"url": url, "title": title, "status": "timeout", "local_path": None})
        except Exception as e:
            logger.warning(f"[download_media_images] Failed {url}: {e}")
            results.append({"url": url, "title": title, "status": f"error: {e}", "local_path": None})

    # Write back manifest with local_path fields added
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[download_media_images] Failed to update manifest: {e}")

    summary = f"Downloaded {downloaded}/{min(len(entries), _MAX_IMAGES)} images to {images_dir}\n"
    summary += json.dumps(results, indent=2)
    return summary
