"""
scripts/test_setup.py
=====================
Setup verification script for RAG AI Teacher Assistant (Module 1).

Verifies:
1. Cross-platform path configuration (config.py)
2. Automatic creation and validation of all essential data directories
3. Proper functioning of core utilities (utils.py)
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_DIR,
    VIDEO_DIR,
    AUDIO_DIR,
    TRANSCRIPT_DIR,
    CHUNK_DIR,
    EMBEDDING_DIR,
    ALL_DATA_DIRS,
    get_config_summary,
)
from src.utils import (
    ensure_dir,
    dir_exists,
    file_exists,
    sanitize_filename,
    create_project_directories,
)


def test_directory_setup() -> bool:
    """
    Creates and verifies all required project directories.

    Returns:
        bool: True if all directories exist and are accessible, False otherwise.
    """
    print("\n[1/3] Verifying and Creating Project Directories...")
    verified = create_project_directories()

    all_ok = True
    for directory in ALL_DATA_DIRS:
        exists = dir_exists(directory)
        status = "OK" if exists else "FAILED"
        print(f"  [{status}] {directory.relative_to(BASE_DIR)} -> {directory}")
        if not exists:
            all_ok = False

    return all_ok


def test_utility_functions() -> bool:
    """
    Validates utility helpers: sanitize_filename, file_exists, dir_exists.

    Returns:
        bool: True if utilities pass test assertions, False otherwise.
    """
    print("\n[2/3] Verifying Utility Functions...")

    # Test 1: sanitize_filename with invalid characters across OSes
    test_cases = [
        ("Lecture 01: Intro to Python? (Part 1).mp4", "Lecture_01_Intro_to_Python_Part_1.mp4"),
        ("Deep/Learning\\Neural*Networks:101.mov", "Deep_Learning_Neural_Networks_101.mov"),
        ("   spaces and ... dots ...   ", "spaces_and_dots"),
        ("", "untitled"),
    ]

    all_utils_ok = True
    for raw_input, expected in test_cases:
        result = sanitize_filename(raw_input)
        if result == expected:
            print(f"  [OK] sanitize_filename('{raw_input}') -> '{result}'")
        else:
            print(f"  [FAILED] sanitize_filename('{raw_input}'): got '{result}', expected '{expected}'")
            all_utils_ok = False

    # Test 2: file_exists and dir_exists
    base_exists = dir_exists(BASE_DIR)
    config_file = BASE_DIR / "src" / "config.py"
    config_exists = file_exists(config_file)

    if base_exists and config_exists:
        print(f"  [OK] dir_exists(BASE_DIR): {base_exists}")
        print(f"  [OK] file_exists(src/config.py): {config_exists}")
    else:
        print(f"  [FAILED] Existence checks failed: base_exists={base_exists}, config_exists={config_exists}")
        all_utils_ok = False

    return all_utils_ok


def test_configuration_summary() -> bool:
    """
    Displays active configuration summary.

    Returns:
        bool: True if configuration summary dictionary is populated.
    """
    print("\n[3/3] Inspecting Configuration Summary...")
    summary = get_config_summary()
    for key, value in summary.items():
        print(f"  - {key}: {value}")

    return len(summary) > 0


def run_all_checks() -> int:
    """
    Executes all setup tests and reports overall status.

    Returns:
        int: 0 if all tests pass, 1 if any test fails.
    """
    print("=" * 60)
    print(" RAG AI Teacher Assistant - Module 1 Setup Verification")
    print("=" * 60)

    dirs_ok = test_directory_setup()
    utils_ok = test_utility_functions()
    config_ok = test_configuration_summary()

    print("\n" + "=" * 60)
    if dirs_ok and utils_ok and config_ok:
        print(" SUCCESS: Module 1 architecture and setup verified successfully!")
        print(" All directories and configuration modules are operational.")
        print("=" * 60)
        return 0
    else:
        print(" FAILURE: One or more setup checks failed. Review errors above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = run_all_checks()
    sys.exit(exit_code)
