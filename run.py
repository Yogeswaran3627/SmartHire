"""
SmartHire Self-Bootstrapping Application Launcher.
Simply open this file and click the Run button (▶️) in VS Code!

Features:
- Automatically verifies all required dependencies (scikit-learn, streamlit, pandas, etc.).
- If any package is missing, it installs them automatically via pip before starting.
- Automatically sets up Streamlit configuration so it NEVER asks for an email.
- Launches the web portal directly in your default browser.
"""
import sys
import subprocess
import importlib.util
from pathlib import Path

# Fix Windows console encoding
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Mapping of Python import names to pip distribution names
REQUIRED_MODULES = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "pypdf": "pypdf",
    "docx": "python-docx",
    "joblib": "joblib",
    "openpyxl": "openpyxl",
}

def ensure_environment():
    """Checks for required dependencies and installs any missing ones automatically."""
    root_dir = Path(__file__).resolve().parent
    req_file = root_dir / "requirements.txt"

    # 1. Check for missing packages
    missing_packages = []
    for import_name, pip_name in REQUIRED_MODULES.items():
        if importlib.util.find_spec(import_name) is None:
            missing_packages.append(pip_name)

    # 2. Install missing packages if needed
    if missing_packages:
        print("=" * 65)
        print("[SmartHire] Missing packages detected on this system:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstalling missing dependencies automatically... Please wait.")
        print("=" * 65)

        try:
            if req_file.exists():
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("\n[SmartHire] All dependencies installed successfully!\n")
        except subprocess.CalledProcessError as e:
            print(f"\n[Error] Failed to install dependencies automatically: {e}")
            print("Please try running: pip install -r requirements.txt")
            sys.exit(1)

    # 3. Suppress Streamlit first-run email prompt automatically
    st_dir = root_dir / ".streamlit"
    st_dir.mkdir(exist_ok=True)
    cred_file = st_dir / "credentials.toml"
    if not cred_file.exists():
        cred_file.write_text('[general]\nemail = ""\n', encoding="utf-8")

    cfg_file = st_dir / "config.toml"
    if not cfg_file.exists():
        cfg_file.write_text('[browser]\ngatherUsageStats = false\n[server]\nheadless = false\n', encoding="utf-8")

if __name__ == "__main__":
    # Ensure environment is ready
    ensure_environment()

    # Import streamlit cli after environment is guaranteed
    # pyrefly: ignore [missing-import]
    from streamlit.web import cli as stcli

    app_path = Path(__file__).resolve().parent / "app" / "streamlit_app.py"

    print("=" * 65)
    print("[*] Launching SmartHire Web Application...")
    print("=" * 65)

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--browser.gatherUsageStats=false"
    ]
    sys.exit(stcli.main())

