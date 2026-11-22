"""Install SpeakNow from this checkout for the current Windows user."""

import argparse
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import shutil
import subprocess
import sys


SOURCE = Path(__file__).resolve().parent


def add_to_user_path(directory: Path) -> bool:
    import winreg

    target = os.path.normcase(os.path.normpath(directory))
    with winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER, "Environment", access=winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE
    ) as environment:
        try:
            path, value_type = winreg.QueryValueEx(environment, "Path")
        except FileNotFoundError:
            path, value_type = "", winreg.REG_EXPAND_SZ
        entries = path.split(";")
        for entry in entries:
            expanded = os.path.expandvars(entry.strip().strip('"'))
            if expanded and os.path.normcase(os.path.normpath(expanded)) == target:
                return False
        new_path = f"{path.rstrip(';')};{directory}" if path.rstrip(";") else str(directory)
        winreg.SetValueEx(environment, "Path", 0, value_type, new_path)

    # Let Explorer refresh the user environment for newly opened applications.
    notify = ctypes.windll.user32.SendMessageTimeoutW
    notify.argtypes = (
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPCWSTR,
        wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t),
    )
    notify.restype = wintypes.LPARAM
    notify(0xFFFF, 0x001A, 0, "Environment", 0x0002, 1000, None)
    return True


def create_shortcuts(launcher: Path, icon: Path, folders: list[str]) -> None:
    powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    environment = {
        **os.environ,
        "SPEAKNOW_INSTALL_LAUNCHER": str(launcher),
        "SPEAKNOW_INSTALL_ICON": str(icon),
        "SPEAKNOW_INSTALL_FOLDERS": ",".join(folders),
    }
    script = """
        $ErrorActionPreference = 'Stop'
        $shell = New-Object -ComObject WScript.Shell
        foreach ($folder in $env:SPEAKNOW_INSTALL_FOLDERS.Split(',')) {
            $destination = $shell.SpecialFolders.Item($folder)
            if (-not $destination) { throw "Could not locate the Windows $folder folder." }
            $path = Join-Path $destination 'SpeakNow.lnk'
            $shortcut = $shell.CreateShortcut($path)
            $shortcut.TargetPath = $env:SPEAKNOW_INSTALL_LAUNCHER
            $shortcut.WorkingDirectory = Split-Path -Parent $env:SPEAKNOW_INSTALL_LAUNCHER
            $shortcut.IconLocation = "$env:SPEAKNOW_INSTALL_ICON,0"
            $shortcut.Description = 'SpeakNow voice assistant'
            $shortcut.Save()
            Write-Output $path
        }
    """
    subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        env=environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--add-to-path", action="store_true", help="Add SpeakNow to the current user's PATH")
    parser.add_argument("--desktop", action="store_true", help="Create a desktop shortcut")
    parser.add_argument("--start-menu", action="store_true", help="Create a Start menu shortcut")
    parser.add_argument("--all", action="store_true", help="Enable all three options")
    parser.add_argument("--install-dir", type=Path, help=r"Install location (default: %%LOCALAPPDATA%%\Programs\SpeakNow)")
    options = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This installer runs on Windows")
    if sys.version_info < (3, 11):
        parser.error("SpeakNow requires Python 3.11 or newer")

    install_dir = options.install_dir or Path(os.environ["LOCALAPPDATA"]) / "Programs" / "SpeakNow"
    install_dir = install_dir.resolve()
    venv = install_dir / "venv"
    scripts = venv / "Scripts"
    python = scripts / "python.exe"
    launcher = scripts / "speaknow.exe"
    icon = install_dir / "speaknow.ico"

    if not python.is_file() or not (scripts / "pip.exe").is_file():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", str(SOURCE)], check=True)
    if not launcher.is_file():
        raise FileNotFoundError(f"SpeakNow launcher was not installed at {launcher}")
    shutil.copyfile(SOURCE / "images" / "speaknow.ico", icon)

    folders = []
    if options.desktop or options.all:
        folders.append("Desktop")
    if options.start_menu or options.all:
        folders.append("Programs")
    if folders:
        create_shortcuts(launcher, icon, folders)
    if options.add_to_path or options.all:
        if add_to_user_path(scripts):
            print("Added SpeakNow to your PATH. Open a new terminal to use it.")
        else:
            print("SpeakNow is already on your PATH.")
    print(f"Installed SpeakNow at {launcher}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"SpeakNow installation failed. {error}")
