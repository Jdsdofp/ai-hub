# app/projects/epi_check/api/ws_proxy.py
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from loguru import logger

try:
    import websockets
    import websockets.exceptions
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    logger.warning("[WsProxy] 'websockets' nao instalado. Execute: pip install websockets --break-system-packages")

router = APIRouter()


@router.websocket("/ws/lock")
async def lock_ws_proxy(
    websocket: WebSocket,
    lock_ip: str = Query(default="192.168.68.100"),
):
    await websocket.accept()

    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(f"[WsProxy] Conectado: {client_host} -> ESP32 ws://{lock_ip}:81")

    if not HAS_WEBSOCKETS:
        await websocket.send_text(json.dumps({"event": "proxy_error", "error": "websockets nao instalado"}))
        await websocket.close(code=1011)
        return

    esp_url = f"ws://{lock_ip}:81"

    try:
        async with websockets.connect(esp_url, ping_interval=20, ping_timeout=10, close_timeout=5) as esp_ws:
            logger.info(f"[WsProxy] Conectado ao ESP32: {esp_url}")
            await websocket.send_text(json.dumps({"event": "proxy_connected", "esp_url": esp_url, "lock_ip": lock_ip}))

            async def frontend_to_esp():
                try:
                    while True:
                        data = await websocket.receive_text()
                        await esp_ws.send(data)
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    logger.warning(f"[WsProxy] frontend_to_esp: {e}")

            async def esp_to_frontend():
                try:
                    async for message in esp_ws:
                        await websocket.send_text(message)
                except websockets.exceptions.ConnectionClosed as e:
                    try:
                        await websocket.send_text(json.dumps({"event": "proxy_esp_disconnected", "reason": str(e)}))
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"[WsProxy] esp_to_frontend: {e}")

            done, pending = await asyncio.wait(
                [asyncio.create_task(frontend_to_esp()), asyncio.create_task(esp_to_frontend())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    except OSError as e:
        logger.error(f"[WsProxy] ESP32 inacessivel {esp_url}: {e}")
        try:
            await websocket.send_text(json.dumps({"event": "proxy_error", "error": f"ESP32 inacessivel em {esp_url}", "detail": str(e)}))
        except Exception:
            pass
        await websocket.close(code=1011)
    except Exception as e:
        logger.error(f"[WsProxy] Erro: {e}")
        await websocket.close(code=1011)
    finally:
        logger.info(f"[WsProxy] Sessao encerrada: {client_host}")
