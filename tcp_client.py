import socket
import sys
from types import resolve_bases

TCP_PORT = 5500
ad = input("IP address:")

tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

tcp_sock.connect((ad, TCP_PORT))

# 키보드로부터문자열을입력받아서서버로전송
while True:
  data = input("Message: ")
  tcp_sock.sendall(data.encode())

  res = tcp_sock.recv(1024)
  print("recieved data :{}".format(res.decode()))

  if data == 'x':
    tcp_sock.close()
    break

sys.exit()
