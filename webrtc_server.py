"""
Air Guitar Pro - WebRTC Server
Handles P2P connection between PC game and mobile controller using aiortc.
"""

import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store for peer connections and signaling
pcs = set()


def create_pc():
    """Create a new RTCPeerConnection."""
    pc = RTCPeerConnection()
    pcs.add(pc)

    # @pc.on("connectionstatechange")
    # async def on_connectionstatechange():
    #     logger.info(f"Connection state: {pc.connectionState}")
    #     if pc.connectionState == "failed":
    #         await pc.close()
    #         pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")
        # @track.on("ended")
        # async def on_ended():
        #     logger.info(f"Track ended: {track.kind}")

    return pc


async def offer_handler(request):
    """Handle WebRTC offer from mobile client."""
    params = await request.json()

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = create_pc()

    # Handle data channel
    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"Data channel created: {channel.label}")

        @channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
                if data.get("type") == "FRET_UPDATE":
                    # Send to game via callback or shared state
                    fret_states = data.get("payload", [0, 0, 0, 0, 0, 0])
                    logger.info(f"FRET update: {fret_states}")
                    # TODO: Send to game instance
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {message}")

    # Set remote description
    await pc.setRemoteDescription(offer)

    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


async def index_handler(request):
    """Serve the main page."""
    # Redirect to game
    return web.Response(text="Air Guitar Pro WebRTC Server. Use /mobile for controller.")


async def mobile_handler(request):
    """Serve the mobile controller page."""
    html_path = Path(__file__).parent / "templates" / "mobile_webrtc.html"
    if html_path.exists():
        return web.Response(text=html_path.read_text(), content_type="text/html")
    return web.Response(text="Mobile controller not found.", status=404)


async def on_shutdown(app):
    """Cleanup on shutdown."""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def web_server():
    """Start the WebRTC signaling server."""
    app = web.Application()

    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/", index_handler)
    app.router.add_get("/mobile", mobile_handler)
    app.router.add_post("/offer", offer_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()

    logger.info("WebRTC server running on http://0.0.0.0:8080")

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(web_server())
