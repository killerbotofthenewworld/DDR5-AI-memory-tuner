import subprocess
import webbrowser
import time

# Start Streamlit in the background
subprocess.Popen([
	"python", "-m", "streamlit", "run",
	"src/web_interface/main.py",
	"--server.port", "8521",
])

# Wait a few seconds for the server to start
time.sleep(3)

# Open the web interface in the default browser
webbrowser.open("http://localhost:8521")
