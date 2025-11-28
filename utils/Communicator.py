import pickle
import random
import struct
import socket
import logging
import time
from typing import List, Tuple


class Communicator(object):
    def __init__(self, index=None, ip_address=None):
        # self.logger 将由子类 (Client, Server) 定义
        self.index = index
        self.key = ip_address
        self.sock = socket.socket()

    def send_msg(self, sock, msg, msg_type=None):
        if not hasattr(self, 'logger'):
             # 创建一个备用logger，以防子类没有定义
            self.logger = logging.getLogger(self.__class__.__name__)

        if msg_type is not None:
            msg = (msg_type, msg)

        msg_pickle = pickle.dumps(msg)
        sock.sendall(struct.pack(">I", len(msg_pickle)))
        sock.sendall(msg_pickle)

        # 【修正】使用 self.logger
        if isinstance(msg, tuple):
            self.logger.debug(f"Sent [{msg[0]}] to {sock.getpeername()}")
        else:
            self.logger.debug(f"Sent [dict] to {sock.getpeername()}")
        return True

    def recv_msg(self, sock, expect_msg_type=None, timeout=None):
        """
        【修正后】的接收函数，返回值统一为 (msg_type, content) 元组。
        成功时，返回消息类型和内容。
        失败或连接关闭时，返回 (None, None)。
        """
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(self.__class__.__name__)
        
        if timeout: sock.settimeout(timeout)
        try:
            raw_msglen = self._recvall(sock, 4)
            if not raw_msglen:
                self.logger.warning(f"Connection with {sock.getpeername()} seems to be closed (received empty header).")
                return None, None

            msg_len = struct.unpack(">I", raw_msglen)[0]
            msg_pickle = self._recvall(sock, msg_len)
            if not msg_pickle:
                self.logger.warning(f"Connection with {sock.getpeername()} closed while receiving message body.")
                return None, None

            msg = pickle.loads(msg_pickle)
            msg_type, content = None, None

            if isinstance(msg, tuple) and len(msg) == 2:
                msg_type, content = msg
                # 【修正】使用 self.logger
                self.logger.debug(f"Received [{msg_type}] from {sock.getpeername()}")
            else:
                msg_type = "UNKNOWN"
                content = msg
                # 【修正】使用 self.logger
                self.logger.debug(f"Received non-standard message format from {sock.getpeername()}.")

            if expect_msg_type is not None and msg_type != expect_msg_type:
                # 【修正】使用 self.logger
                self.logger.error(f"Expected message type '{expect_msg_type}' but received '{msg_type}'. Discarding message.")
                return msg_type, None

            return msg_type, content

        except (ConnectionResetError, BrokenPipeError, struct.error) as e:
            # 【修正】使用 self.logger
            self.logger.error(f"Connection error with {sock.getpeername()}: {e}")
            return None, None
        except Exception as e:
            # 【修正】使用 self.logger
            self.logger.critical(f"An unexpected error occurred during recv_msg: {e}", exc_info=True)
            return None, None
        finally:
            if timeout: sock.settimeout(None)

    def _recvall(self, sock, n):
        data = b''
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data += packet
            except BlockingIOError:
                time.sleep(0.001)
                continue
        return data


def connect_with_retry(server_addresses: List[Tuple[str, int]], logger_ref: logging.Logger, max_retries=15, initial_delay=1, max_delay=10):
    log = logger_ref if logger_ref else logging.getLogger("connect_with_retry")
    delay = initial_delay
    
    for attempt in range(max_retries):
        server_addr = random.choice(server_addresses)
        try:
            log.debug(f"Attempt {attempt + 1}/{max_retries}: trying to connect to {server_addr}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)  # 连接本身的超时可以短一些
            sock.connect(server_addr)
            sock.settimeout(None) # 连接成功后取消超时
            return sock
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            log.debug(f"Failed to connect to {server_addr} ({type(e).__name__}). Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            # 指数增加延迟时间，但不超过最大值
            delay = min(delay * 1.5, max_delay)
    
    log.error(f"Failed to connect to any server after {max_retries} attempts.")
    return None

def get_clients_info(client_num, server_addr, logger_ref=None):
    """
    【修正后】的独立函数，必须通过参数接收logger。
    """
    log = logger_ref if logger_ref else logging.getLogger("get_clients_info")

    self_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self_sock.bind(server_addr)

    client_sockets_dict = {}
    clients_key_list = []

    self_sock.listen(client_num + 10)
    while len(client_sockets_dict) < client_num:
        client_socket, (client_ip, client_port) = self_sock.accept()
        log.info('Got connection from:' + str(client_ip) + ":" + str(client_port))

        client_key = str(client_ip) + ":" + str(client_port)
        clients_key_list.append(client_key)
        client_sockets_dict[client_key] = client_socket

        log.info('len(client_sockets):', len(client_sockets_dict))

    return clients_key_list, client_sockets_dict, self_sock