import os
import re
from typing import Optional

# Try to load .env using python-dotenv if it's installed. This is opportunistic:
# failure to import dotenv is non-fatal and we simply fall back to existing os.environ.
try:
    from dotenv import find_dotenv, load_dotenv

    _dotenv_path = find_dotenv()  # searches upwards for a .env file
    if _dotenv_path:
        # load into os.environ without overriding existing env vars
        load_dotenv(_dotenv_path, override=False)
except Exception:
    # ignore any errors and continue using os.environ
    pass

MIN_VERSION = "8.0.23"
PKG_NAME = "mysql-connector-python"


def _env(name: str) -> Optional[str]:
    """
    Resolve a configuration value by checking real environment variables (os.environ).
    """
    return os.getenv(name)


def _get_config():
    host = _env("DB_HOST")
    port = int(_env("DB_PORT") or 3306)
    user = _env("DB_USER")
    password = _env("DB_PASS")
    database = _env("DB_NAME")
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
        # helpful defaults
        "charset": "utf8mb4",
        "use_unicode": True,
    }


def _parse_version_to_tuple(v: str):
    """
    Convert version string like '8.0.23' or '8.0.23a' to tuple of ints (8,0,23).
    Non-numeric suffixes are ignored. This is a simple comparator sufficient for
    comparing major.minor.patch for typical mysql-connector-python versions.
    """
    # Keep only the numeric parts at start like 8.0.23 from 8.0.23a
    m = re.match(r"^([0-9]+(?:\.[0-9]+)*)", v)
    if not m:
        return ()
    parts = m.group(1).split(".")
    try:
        return tuple(int(x) for x in parts)
    except ValueError:
        # fallback to naive parsing
        return tuple(int(re.sub(r"\D", "", x) or 0) for x in parts)


def _get_installed_package_version(pkg_name: str) -> Optional[str]:
    """
    Try importlib.metadata (py3.8+) then pkg_resources; return version string or None.
    """
    try:
        # Python 3.8+
        from importlib import metadata

        return metadata.version(pkg_name)
    except Exception:
        try:
            # setuptools pkg_resources fallback
            import pkg_resources

            return pkg_resources.get_distribution(pkg_name).version
        except Exception:
            return None


def _ensure_connector_present_and_ok():
    ver = _get_installed_package_version(PKG_NAME)
    if ver is None:
        raise ImportError(
            f"'{PKG_NAME}' is not installed. Install it with:\n\n"
            f"    pip install \"{PKG_NAME}>={MIN_VERSION}\"\n"
        )
    inst = _parse_version_to_tuple(ver)
    required = _parse_version_to_tuple(MIN_VERSION)
    if not inst or inst < required:
        raise ImportError(
            f"Installed '{PKG_NAME}' version is {ver!r} but this project requires >= {MIN_VERSION}.\n"
            f"Please upgrade with:\n\n"
            f"    pip install --upgrade \"{PKG_NAME}>={MIN_VERSION}\"\n"
        )
    # If we reach here, version is acceptable


def get_connection():
    """
    Return a new mysql.connector connection object.

    Raises ImportError if mysql-connector-python is missing or too old.
    """
    # verify package is installed and at least MIN_VERSION
    _ensure_connector_present_and_ok()

    try:
        import mysql.connector
    except Exception as exc:
        raise ImportError(
            "Unable to import mysql.connector even though the package appeared to be installed.\n"
            "Ensure your Python environment is correct and that mysql-connector-python is installed.\n\n"
            f"Original error: {exc}"
        ) from exc

    cfg = _get_config()
    try:
        conn = mysql.connector.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            charset=cfg["charset"],
            use_unicode=cfg["use_unicode"],
            connection_timeout=30,
        )
        return conn
    except mysql.connector.Error as e:
        raise RuntimeError(
            "Failed to connect to MySQL. Please check your DB_* env vars and that the "
            "MySQL server is reachable. Original error: " + str(e)
        ) from e


# convenience export
__all__ = ["get_connection"]
