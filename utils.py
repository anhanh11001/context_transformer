from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent


def print_line_divider():
    print("****************************************************")
