"""Launch the inference server."""

import os
import sys

from sglang.srt.server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

import nacos
import configparser
from urllib.parse import urlparse

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    def register_service(client,service_name,service_ip,service_port,cluster_name,health_check_interval,weight):
        try:
            metadata = {"model":server_args.served_model_name,"port":server_args.port}
            response = client.add_naming_instance(
                service_name,
                service_ip,
                service_port,
                cluster_name,
                weight,
                metadata,
                enable=True,
                healthy=True,
                ephemeral=True,
                heartbeat_interval=health_check_interval
            )
            return response
        except Exception as e:
            print(f"Error registering service to Nacos: {e}")
            return False
    # 创建配置解析器
    config = configparser.ConfigParser()
    # 读取配置文件
    if not config.read('config.ini'):
        sys.exit(0)
    # Nacos server and other configurations
    NACOS_SERVER = config['NACOS']['nacos_server']
    NAMESPACE = config['NACOS']['namespace']
    CLUSTER_NAME = config['NACOS']['cluster_name']
    client = nacos.NacosClient(NACOS_SERVER, namespace=NAMESPACE, username=config['NACOS']['username'], password=config['NACOS']['password'])
    SERVICE_NAME = config['NACOS']['service_name']
    HEALTH_CHECK_INTERVAL = int(config['NACOS']['health_check_interval'])

    # Parse AutoDLServiceURL
    autodl_url = os.environ.get('AutoDLServiceURL')
    AutoDLContainerUUID = os.environ.get('AutoDLContainerUUID')
    if not autodl_url:
        print("Error: AutoDLServiceURL environment variable is not set.")
        sys.exit(0)

    parsed_url = urlparse(autodl_url)
    SERVICE_IP = parsed_url.hostname
    SERVICE_PORT = parsed_url.port
    WEIGHT = 1
    if not SERVICE_IP or not SERVICE_PORT:
        print("Error: Invalid AutoDLServiceURL format.")
        sys.exit(0)

    print(f"Service will be registered with IP: {SERVICE_IP} and Port: {SERVICE_PORT}")
    if not register_service(client,SERVICE_NAME,SERVICE_IP,SERVICE_PORT,CLUSTER_NAME,HEALTH_CHECK_INTERVAL,WEIGHT):
        print("Service is healthy but failed to register.")
        sys.exit(0)
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
