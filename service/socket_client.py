import socket
import struct
import sys
import time
import random
import re
from datetime import datetime

if __name__=="__main__":
  global f
  global index_time

  host = 'localhost'
  port = 8220
  address = (host, port) #Initializing the port and the host for the connection
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.connect(address)
