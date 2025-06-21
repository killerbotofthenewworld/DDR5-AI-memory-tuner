#!/bin/bash

# Prompt for password using Zenity
echo "Requesting password for elevated privileges..."
PASSWORD=$(zenity --password --title="Enter Password for DDR5 Simulator Restart")

if [ -z "$PASSWORD" ]; then
  zenity --error --text="Password not entered. Aborting."
  exit 1
fi

# Restart the DDR5 Simulator
(echo $PASSWORD | sudo -S pkill -f streamlit && streamlit run main.py) &

if [ $? -eq 0 ]; then
  zenity --info --text="DDR5 Simulator restarted successfully."
else
  zenity --error --text="Failed to restart DDR5 Simulator."
fi
