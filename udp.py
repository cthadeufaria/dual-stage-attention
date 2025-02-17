import socket
import struct
import asyncio
import opuslib


class UDPClient:
    def __init__(self, multicast_group, port=12345):
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_socket.bind(('', self.port))
        group = socket.inet_aton(multicast_group)
        mreq = struct.pack('4sL', group, socket.INADDR_ANY)
        self.client_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.client_socket.setblocking(False)
        self.BUFFER_SIZE = 1024 # TODO: adjust video parameters.
        self.SAMPLE_RATE = 48000
        self.CHANNELS = 1
        self.CHUNK_SIZE = 960
        self.decoder = opuslib.Decoder(self.SAMPLE_RATE, self.CHANNELS)

    async def listen(self):
        loop = asyncio.get_event_loop()
        print("Listening to UDP messages...")

        while True:
            try:
                encoded = await loop.sock_recv(self.client_socket, self.BUFFER_SIZE)
                decoded = self.decoder.decode(encoded, self.CHUNK_SIZE)
                print("Decoded message:", decoded)
                await asyncio.sleep(0.001)
            
            except BlockingIOError: # TODO: test if these two are the correct exceptions.
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
