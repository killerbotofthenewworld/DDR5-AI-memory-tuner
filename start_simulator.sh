#!/bin/bash

# Ensure the correct PATH for desktop launches
export PATH="$PATH:$HOME/.local/bin"

# Prompt for password using Zenity (for hardware scanning, not for Streamlit)
echo "Requesting password for hardware scanning..."
PASSWORD=$(zenity --password --title="Enter Password for DDR5 Hardware Scan (if needed)")

if [ -z "$PASSWORD" ]; then
  zenity --error --text="Password not entered. Aborting."
  exit 1
fi

# Kill any running Streamlit processes (ignore errors if not running)
pkill -f streamlit 2>/dev/null

# Start the DDR5 Simulator in the background and open the browser (no sudo)
/usr/bin/env streamlit run main.py &
sleep 3
xdg-open http://localhost:8501 &

# Script ends here; do not attempt to wait for or auto-close Streamlit
