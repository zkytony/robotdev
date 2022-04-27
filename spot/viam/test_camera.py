import asyncio
import logging

from viam.rpc.server import Server
from camera_component import SpotCamera

async def run():
    my_camera = SpotCamera('frontleft')
    server = Server(components=[
        my_camera,
    ])
    await server.serve(log_level=logging.DEBUG)

if __name__ == '__main__':
    asyncio.run(run())
