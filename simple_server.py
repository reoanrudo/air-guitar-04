"""
Simple Flask server for mobile controller.
"""

from flask import Flask, request
from pathlib import Path

app = Flask(__name__)

@app.route('/')
def index():
    ip = request.remote_addr
    return f"""
    <h1>Air Guitar Pro Server</h1>
    <p>Your IP: {ip}</p>
    <p>Mobile controller: <a href="/mobile">/mobile</a></p>
    """

@app.route('/mobile')
def mobile():
    html_path = Path(__file__).parent / "templates" / "mobile_webrtc.html"
    if html_path.exists():
        return html_path.read_text()
    return "Mobile controller not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=False)
