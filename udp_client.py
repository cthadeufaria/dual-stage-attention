import socket, cv2, time


class UDPClient:
    def __init__(self, server_ip="127.0.0.1", server_port=5000):
        self.address = (server_ip, server_port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.BUFFER_SIZE)

    def send(self, message):
        self.socket.sendto(message, self.address)

    def send_video(self, video_path):  # TODO: Include QoS features.
        frame_count = 0
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video.")

            return

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])

            print(f"Sending frame {frame_count}: {buffer.shape}")

            self.send(buffer.tobytes())

            time.sleep(1/32)  # Simulate 32 FPS to be divisible by the fast pathway input size.


def main():
    client = UDPClient()

    path = "/home/dev/repos/dual-stage-attention/datasets/LIVE_NFLX_Plus/assets_mp4_individual/AirShow_HuangBufferBasedAdaptor_Trace_0.mp4"
    client.send_video(path)


if __name__ == "__main__":
    main()