"""
Air Guitar Pro - WebRTC Signaling Server
Handles signaling between PC game and mobile controller for P2P connection.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Port from environment variable (Render) or default
PORT = int(os.environ.get("PORT", 8080))
logger.info(f"Starting signaling server on PORT: {PORT}")

# Store for peer connections and WebSocket clients
pcs = set()
pc_connections: Dict[str, RTCPeerConnection] = {}
websocket_clients: Dict[str, web.WebSocketResponse] = {}


def create_pc(pc_id: str) -> RTCPeerConnection:
    """Create a new RTCPeerConnection with ID."""
    pc = RTCPeerConnection()
    pc_connections[pc_id] = pc
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"PC {pc_id} connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            pc_connections.pop(pc_id, None)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"PC {pc_id} ICE connection state: {pc.iceConnectionState}")

    return pc


async def register_client(request):
    """Register a client (PC or mobile). Returns client ID."""
    client_id = request.match_info.get('client_id', 'unknown')
    client_type = request.query.get('type', 'unknown')  # 'pc' or 'mobile'

    logger.info(f"Client registered: {client_id} ({client_type})")
    return web.json_response({"client_id": client_id, "status": "registered"})


async def websocket_handler(request):
    """WebSocket handler for real-time signaling."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    client_id = None

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get("type")
                sender_id = data.get("sender_id")

                if not client_id and sender_id:
                    client_id = sender_id
                    websocket_clients[client_id] = ws
                    logger.info(f"WebSocket client connected: {client_id}")

                if msg_type == "offer":
                    # Mobile sends offer, forward to PC
                    target_id = data.get("target_id", "pc_game")
                    logger.info(f"Offer from {sender_id} to {target_id}")

                    if target_id in websocket_clients:
                        await websocket_clients[target_id].send_json(data)

                elif msg_type == "answer":
                    # PC sends answer, forward to mobile
                    target_id = data.get("target_id")
                    logger.info(f"Answer from {sender_id} to {target_id}")

                    if target_id in websocket_clients:
                        await websocket_clients[target_id].send_json(data)

                elif msg_type == "ice_candidate":
                    # Forward ICE candidates
                    target_id = data.get("target_id")
                    logger.info(f"ICE candidate from {sender_id} to {target_id}")

                    if target_id in websocket_clients:
                        await websocket_clients[target_id].send_json(data)

                elif msg_type == "pc_connected":
                    # PC is ready and waiting for mobile
                    logger.info(f"PC ready: {sender_id}")
                    # Broadcast to all mobile clients
                    for cid, client_ws in websocket_clients.items():
                        if cid != sender_id:
                            await client_ws.send_json({
                                "type": "pc_ready",
                                "pc_id": sender_id
                            })

            elif msg.type == WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if client_id:
            websocket_clients.pop(client_id, None)
            logger.info(f"WebSocket client disconnected: {client_id}")

    return ws


async def index_handler(request):
    """Serve the main page."""
    return web.Response(text="Air Guitar Pro WebRTC Signaling Server. Use /mobile for controller.")


async def mobile_handler(request):
    """Serve the mobile controller page."""
    html_path = Path(__file__).parent / "templates" / "mobile_webrtc.html"
    if html_path.exists():
        return web.Response(text=html_path.read_text(), content_type="text/html")
    return web.Response(text="Mobile controller not found.", status=404)


async def on_shutdown(app):
    """Cleanup on shutdown."""
    # Close all WebSocket connections
    for ws in websocket_clients.values():
        await ws.close()

    # Close all peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    pc_connections.clear()


async def web_server():
    """Start the WebRTC signaling server."""
    app = web.Application()

    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/", index_handler)
    app.router.add_get("/mobile", mobile_handler)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_post("/register/{client_id}", register_client)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()

    logger.info(f"WebRTC signaling server running on http://0.0.0.0:{PORT}")

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(web_server())
