import asyncio
import logging

from viam.rpc.server import Server
from .camera.spot_camera import SpotCamera
from viam.registry import Registry

async def run():
    my_camera = SpotCamera('frontleft')
    server = Server(components=[
        my_camera,
    ])
    print(Registry.REGISTERED_COMPONENTS)
    await server.serve(log_level=logging.DEBUG)

if __name__ == '__main__':
    asyncio.run(run())
