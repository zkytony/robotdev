# Image Services

There are two `.proto` files for image services:

- [image.proto](https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#image-proto)

- [image_service.proto](https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#image-service-proto)

In Python, if you want to request images from specified sources,
you will use the `ImageClient` in `bosdyn.client.image`.

Tip: prefer to use the `build_image_request` function
in combination with the `get_image_async` function
so that you can specify image quality. The former is
a module function in `bosdyn.client.image`:
```python
def build_image_request(image_source_name, quality_percent=75, image_format=None):
    """Helper function which builds an ImageRequest from an image source name.

    By default the robot will choose an appropriate format when no image format
    is provided. For example, it will choose JPEG for visual images, or RAW for
    depth images. Clients can provide an image_format for other cases.

    Args:
        image_source_name (string): The image source to query.
        quality_percent (int): The image quality from [0,100] (percent-value).
        image_format (image_pb2.Image.Format): The type of format for the image
                                               data, such as JPEG, RAW, or RLE.

    Returns:
        The ImageRequest protobuf message for the given parameters.
    """
    ...
```
The `get_image_async` function returns a list of image responses for each of the requested sources.



## On Streaming

[This Spot Support thread](https://support.bostondynamics.com/s/question/0D54X00006d1PJRSA2/is-it-possible-to-intercept-the-camera-rtsp-streams-from-the-onboard-cameras-on-spot-to-put-them-up-on-another-computer-i-know-its-possible-to-have-another-tablet-connected-in-observe-mode-im-just-looking-to-open-the-video-streams-in-vlc)
asked whether you could divert [RTSP stream](https://en.wikipedia.org/wiki/Real_Time_Streaming_Protocol)
to another computer. The support says, "What the Spot app does in order to see
this video stream is repeatedly call the get image function." This
matches my understanding from the Support's email that says they
stream images on-board at 15Hz.
