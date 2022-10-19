import rbd_spot
import matplotlib.pyplot as plt


def main():
    conn = rbd_spot.SpotSDKConn(sdk_name="MyStreamImageClient")
    image_client = rbd_spot.image.create_client(conn)
    while True:
        image_requests = rbd_spot.image.build_image_requests(["frontleft_fisheye_image"])
        result, time_taken = rbd_spot.image.getImage(image_client, image_requests)
        print(time_taken)

        for image_response in result:
            img = rbd_spot.image.imgarray_from_response(image_response, conn)
            plt.imshow(img, cmap='gray')
            plt.show(block=False)
            plt.pause(1)


if __name__ == "__main__":
    main()
