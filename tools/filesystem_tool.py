from langchain.tools import tool
from typing import Optional
import logging
import os
import json

logger = logging.getLogger(__name__)


@tool
def write_file(path: str, content: str) -> str:
    """
    Write text content to a file, creating parent directories as needed.

    Args:
        path: Absolute or relative file path to write
        content: Text content to write

    Returns:
        Confirmation message with the path written
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"[write_file] Wrote {len(content)} chars to {path}")
        return f"File written: {path}"
    except Exception as e:
        logger.error(f"[write_file] Error writing {path}: {e}", exc_info=True)
        return f"Error writing file: {str(e)}"


@tool
def read_file(path: str) -> str:
    """
    Read text content from a file.

    Args:
        path: File path to read

    Returns:
        File contents as a string, or an error message
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"[read_file] Read {len(content)} chars from {path}")
        return content
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as e:
        logger.error(f"[read_file] Error reading {path}: {e}", exc_info=True)
        return f"Error reading file: {str(e)}"


@tool
def append_json_array(path: str, entry: str) -> str:
    """
    Append a JSON object (provided as a string) to a JSON array file.
    Creates the file with an empty array if it does not exist.

    Args:
        path: Path to the JSON file
        entry: JSON string of the object to append

    Returns:
        Confirmation message
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        data.append(json.loads(entry))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"[append_json_array] Appended entry to {path} ({len(data)} total)")
        return f"Entry appended to {path} (total entries: {len(data)})"
    except Exception as e:
        logger.error(f"[append_json_array] Error: {e}", exc_info=True)
        return f"Error appending JSON: {str(e)}"


@tool
def list_files(directory: str) -> str:
    """
    List all files in a directory recursively.

    Args:
        directory: Directory path to list

    Returns:
        Newline-separated list of file paths
    """
    try:
        if not os.path.isdir(directory):
            return f"Error: Not a directory: {directory}"
        result = []
        for root, _, files in os.walk(directory):
            for fname in files:
                result.append(os.path.join(root, fname))
        return "\n".join(result) if result else "Directory is empty."
    except Exception as e:
        return f"Error listing directory: {str(e)}"
