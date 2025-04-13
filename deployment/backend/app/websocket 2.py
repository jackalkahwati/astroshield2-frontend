from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional
import asyncio
import json
from datetime import datetime

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)

    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]

    async def broadcast_to_client(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                try:
                    await connection.send_json(message)
                except WebSocketDisconnect:
                    await self.disconnect(connection, client_id)

    async def broadcast(self, message: dict):
        for client_id in self.active_connections:
            await self.broadcast_to_client(client_id, message)

manager = ConnectionManager()

async def handle_websocket(websocket: WebSocket, client_id: str):
    try:
        await manager.connect(websocket, client_id)
        
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_status",
            "status": "connected",
            "timestamp": datetime.utcnow().isoformat()
        })

        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "subscribe":
                    # Handle subscription requests
                    pass
                elif message.get("type") == "unsubscribe":
                    # Handle unsubscription requests
                    pass
                
            except WebSocketDisconnect:
                manager.disconnect(websocket, client_id)
                await manager.broadcast_to_client(
                    client_id,
                    {
                        "type": "connection_status",
                        "status": "disconnected",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                break
            
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })

    except Exception as e:
        print(f"Error handling WebSocket connection: {e}")

async def send_ccdm_update(client_id: str, data: dict):
    await manager.broadcast_to_client(
        client_id,
        {
            "type": "ccdm_update",
            "payload": data,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def send_thermal_signature(client_id: str, data: dict):
    await manager.broadcast_to_client(
        client_id,
        {
            "type": "thermal_signature",
            "payload": data,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def send_shape_change(client_id: str, data: dict):
    await manager.broadcast_to_client(
        client_id,
        {
            "type": "shape_change",
            "payload": data,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def send_propulsive_event(client_id: str, data: dict):
    await manager.broadcast_to_client(
        client_id,
        {
            "type": "propulsive_event",
            "payload": data,
            "timestamp": datetime.utcnow().isoformat()
        }
    ) 