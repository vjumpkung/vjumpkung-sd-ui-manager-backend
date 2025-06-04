import asyncio
import json
from enum import Enum

from event_handler import manager


class Status(Enum):
    NOT_RUNNING = 0
    RUNNING = 2


class UIPort(Enum):
    INVOKEAI = 9090 # localhost is 9090 but reverse proxy to 3001
    FORGE = 7860
    COMFY = 8188


class ProgramStatus:

    MAP_STATUS = {
        Status.NOT_RUNNING: "NOT_RUNNING",
        Status.RUNNING: "RUNNING",
    }

    MAP_PORT = {
        "INVOKEAI": UIPort.INVOKEAI.value,
        "FORGE": UIPort.FORGE.value,
        "COMFY": UIPort.COMFY.value,
    }

    def __init__(self):
        self.status = Status.NOT_RUNNING

    def get_status(self):
        return self.MAP_STATUS[self.status]

    async def ping_check(self, host="127.0.0.1", port=UIPort.COMFY):
        
        temp = Status.NOT_RUNNING
        
        while True:
            try:
                reader, writer = await asyncio.open_connection(host, port)
                writer.close()
                await writer.wait_closed()

                temp = Status.RUNNING

            except (ConnectionRefusedError, OSError):
                temp = Status.NOT_RUNNING

            if temp != self.status:
                
                self.status = temp
                
                send = {
                    "type": "monitor",
                    "data": {
                        "status": self.MAP_STATUS[self.status],
                    },
                }

                await manager.broadcast(json.dumps(send))
                
            await asyncio.sleep(5)


programStatus = ProgramStatus()
