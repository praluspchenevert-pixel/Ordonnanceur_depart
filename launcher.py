# ============================================================
# launcher.py
# Point d'entree pour construire un executable autonome (PyInstaller)
# de l'application PigOptim. Demarre le serveur Streamlit en interne
# et ouvre automatiquement le navigateur par defaut.
#
# NE PAS lancer avec "streamlit run launcher.py" : ce fichier est
# concu pour etre execute directement (python launcher.py), ou
# empaquete avec PyInstaller. Pour un usage normal en developpement,
# continuez a utiliser : streamlit run pigoptim_app.py
# ============================================================

import os
import sys
import threading
import time
import webbrowser

from streamlit.web import cli as stcli


def _open_browser_later(url: str, delay: float = 2.0):
    time.sleep(delay)
    webbrowser.open(url)


def main():
    # Determine le dossier contenant les fichiers de l'app, que l'on
    # tourne en script normal ou depuis un executable PyInstaller.
    if getattr(sys, "frozen", False):
        base_dir = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(base_dir, "pigoptim_app.py")

    threading.Thread(target=_open_browser_later, args=("http://localhost:8501",), daemon=True).start()

    sys.argv = [
        "streamlit", "run", app_path,
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
