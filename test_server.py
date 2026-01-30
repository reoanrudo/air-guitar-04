"""
Simple HTTP server for mobile controller access test.
"""

from aiohttp import web
from pathlib import Path

async def mobile_handler(request):
    """Serve mobile controller page."""
    html_path = Path(__file__).parent / "templates" / "mobile_webrtc.html"
    if html_path.exists():
        return web.Response(text=html_path.read_text(), content_type="text/html")
    return web.Response(text="Mobile controller not found", status=404)

async def index_handler(request):
    """Index page."""
    ip = request.headers.get('X-Forwarded-For', request.remote)
    return web.Response(text=f"Air Guitar Pro Server\n\nYour IP: {ip}\nMobile controller: /mobile")

async def start_server():
    """Start the server."""
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/mobile", mobile_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8081)
    await site.start()

    print("Server running on http://0.0.0.0:8081")
    print("Mobile: http://<YOUR-IP>:8081/mobile")

    try:
        await web.AppRunner(app).cleanup()
    except:
        pass

if __name__ == "__main__":
    import asyncio
    asyncio.run(start_server())
