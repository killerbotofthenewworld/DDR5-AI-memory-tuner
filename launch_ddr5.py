import subprocess
import webbrowser
import time

# Start Streamlit in the background
subprocess.Popen(["streamlit", "run", "main.py"])

# Wait a few seconds for the server to start
time.sleep(3)

# Open the web interface in the default browser
webbrowser.open("http://localhost:8501")
