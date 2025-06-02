import socket, cv2
import numpy as np


class UDPServer:
    def __init__(self, server_ip="127.0.0.1", server_port=5000):
        self.port = server_port
        self.ip = server_ip
        self.BUFFER_SIZE = 65536
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.BUFFER_SIZE)
        self.socket_addr = (self.ip, self.port)
        self.socket.bind(self.socket_addr)

        print(f"UDP Server listening at {self.ip}:{self.port}")
        print("Host IP is:", socket.gethostbyname(socket.gethostname()))

    def receive(self):
        """Receive data from the UDP socket."""
        try:
            data, _ = self.socket.recvfrom(self.BUFFER_SIZE)
            if data:
                return data
        except socket.error as e:
            print(f"Socket error: {e}")
            return None

    def receive_video(self):
        """Receive video frames from the UDP socket."""
        while True:
            data = self.receive()
            # data = base64.b64decode(data, validate=True)
            data = np.frombuffer(data, np.uint8)

            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("No frame received or decoding failed.")
                continue

            if frame is not None:
                print(f"Received frame: {frame.shape}")


def main():
    server = UDPServer()
    server.receive_video()


if __name__ == "__main__":
    main()