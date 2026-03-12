import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8085
# Serve from 01_dataPreparation so we can access artworks_data_clean.csv
DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
        
    def log_message(self, format, *args):
        # Suppress logging to keep console clean
        pass

def run():
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving dashboard on http://localhost:{PORT}/dashboard/dashboard.html")
            print("Press Ctrl+C to stop the server.")
            
            # Open the web browser automatically
            webbrowser.open(f"http://localhost:{PORT}/dashboard/dashboard.html")
            
            # Serve until stopped
            httpd.serve_forever()
    except OSError:
        print(f"Port {PORT} is already in use. Please try a different port or kill the process.")
    except KeyboardInterrupt:
        print("\nStopping server...")
        sys.exit(0)

if __name__ == "__main__":
    run()
