import asyncio
from viam.rpc.dial import DialOptions, dial_direct
from viam.components.camera import CameraClient

async def client():
    opts = DialOptions(insecure=True)
    async with await dial_direct('localhost:9090', opts) as channel:
        print('\n#### CAMERA ####')
        client = CameraClient('spot-camera-frontleft', channel)
        img = await client.get_frame()
        img.show()
        await asyncio.sleep(1)
        img.close()


if __name__ == '__main__':
    asyncio.run(client())
