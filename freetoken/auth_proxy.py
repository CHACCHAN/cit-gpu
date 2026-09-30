"""Bearer-token check in front of the SGLang Gateway.

The Gateway's control_plane_api_keys only guard its admin API; /v1/* is open, and FreeToken
workers have no key of their own. This proxy is the only listener that Cloudflare Tunnel reaches.
It answers 401 unless the request carries the API key, passes only /v1/* and /health through,
and streams responses (SSE) without buffering. Nothing about a request is logged.
"""

import hmac
import os

import aiohttp
from aiohttp import web

KEY = os.environ["FT_API_KEY"].encode()
UPSTREAM = f"http://127.0.0.1:{os.environ.get('FT_GATEWAY_PORT', '5049')}"
# Hop-by-hop headers, plus the client's key (the Gateway does not need it) and Host.
DROP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
        "transfer-encoding", "upgrade", "authorization", "host", "content-length", "accept-encoding"}


def authorized(request):
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode(), KEY)


async def handle(request):
    if not authorized(request):
        return web.json_response({"error": {"message": "Unauthorized", "type": "invalid_api_key"}},
                                 status=401, headers={"WWW-Authenticate": "Bearer"})
    if request.path != "/health" and not request.path.startswith("/v1/"):
        return web.json_response({"error": {"message": "Not found"}}, status=404)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP}
    body = await request.read()
    try:
        async with request.app["session"].request(
                request.method, UPSTREAM + request.path_qs, headers=headers, data=body or None) as upstream:
            response = web.StreamResponse(status=upstream.status, reason=upstream.reason)
            for name, value in upstream.headers.items():
                if name.lower() not in DROP and name.lower() != "content-encoding":
                    response.headers[name] = value
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response
    except aiohttp.ClientError:
        return web.json_response({"error": {"message": "Gateway unavailable"}}, status=502)


async def start_session(app):
    # No total timeout: a 260K-token prompt takes minutes to prefill. Idle reads are bounded.
    app["session"] = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=2400),
        auto_decompress=False)


async def close_session(app):
    await app["session"].close()


def main():
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(start_session)
    app.on_cleanup.append(close_session)
    app.router.add_route("*", "/{tail:.*}", handle)
    web.run_app(app, host="127.0.0.1", port=int(os.environ.get("FT_PORT", "5050")),
                access_log=None, print=None)


if __name__ == "__main__":
    main()
