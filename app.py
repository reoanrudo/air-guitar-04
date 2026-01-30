"""
Air Guitar Pro - Flask Application
A two-device rock simulator with WebRTC and HandPose detection.
"""

from flask import Flask, render_template, request
import random
import string

app = Flask(__name__)

# Constants for room ID generation
ROOM_ID_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
ROOM_ID_LENGTH = 4


def generate_room_id() -> str:
    """Generate a random 4-character room ID."""
    return ''.join(random.choices(ROOM_ID_CHARS, k=ROOM_ID_LENGTH))


@app.route('/')
def index():
    """Render the lobby page."""
    # Check if room ID is provided in URL hash (simulated via query param)
    room_id = request.args.get('room', '')
    return render_template('lobby.html', room_id=room_id)


@app.route('/pc')
def pc_player():
    """Render the PC player page."""
    room_id = request.args.get('room', generate_room_id())
    return render_template('pc.html', room_id=room_id)


@app.route('/mobile')
def mobile_controller():
    """Render the mobile controller page."""
    room_id = request.args.get('room', '')
    if not room_id or len(room_id) != ROOM_ID_LENGTH:
        return "Error: Please provide a valid 4-character room ID.", 400
    return render_template('mobile.html', room_id=room_id)


@app.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok', 'app': 'Air Guitar Pro'}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
