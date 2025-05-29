import socket, cv2, imutils, base64


class UDPClient:
    def __init__(self, server_ip="127.0.0.1", server_port=5000):
        self.address = (server_ip, server_port)
        self.BUFFER_SIZE = 65536
        self.FRAME_HEADER_SIZE = 12
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.BUFFER_SIZE)

    def send(self, message):
        self.socket.sendto(message, self.address)

    def send_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=400)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            data = base64.b64encode(buffer)
            self.send(data)


def main():
    client = UDPClient()

    path = "/home/dev/repos/dual-stage-attention/datasets/LIVE_NFLX_Plus/assets_mp4_individual/AirShow_HuangBufferBasedAdaptor_Trace_0.mp4"
    client.send_video(path)




if __name__ == "__main__":
    main()