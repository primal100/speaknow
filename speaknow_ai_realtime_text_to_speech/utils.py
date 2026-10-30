from pathlib import Path


def update_log_config(log_config_file: str | Path = None, base_log_dir: str | Path = None):
    config_path = Path(log_config_file)

    if not config_path.exists():
        print(f"Error: {log_config_file} not found.")
        return

    content = config_path.read_text()

    # 2. Replace the placeholder with the actual variable value
    updated_content = content.replace("%BASE_LOG_DIR%", base_log_dir.as_posix())

    # 3. Write it back (or to a new file)
    config_path.write_text(updated_content)
