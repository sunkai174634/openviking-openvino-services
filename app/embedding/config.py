import os

MODEL_ID = os.environ.get('MODEL_ID', 'qwen3-embedding-0.6b-openvino-int8')
MODEL_DIR = os.environ.get('MODEL_DIR', '/models/OpenVINO/Qwen3-Embedding-0.6B-int8-ov')
MAX_INPUT_TOKENS = int(os.environ.get('MAX_INPUT_TOKENS', '4096'))
OPENVINO_DEVICE = os.environ.get('OPENVINO_DEVICE', 'GPU')
MAX_QUEUE_SIZE = int(os.environ.get('MAX_QUEUE_SIZE', '4'))
QUEUE_TIMEOUT_SECONDS = float(os.environ.get('QUEUE_TIMEOUT_SECONDS', '2'))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get('REQUEST_TIMEOUT_SECONDS', '30'))
LONG_REQUEST_TOKENS = int(os.environ.get('LONG_REQUEST_TOKENS', '2048'))
SHORT_REQUEST_TOKENS = int(os.environ.get('SHORT_REQUEST_TOKENS', '256'))
LONG_QUEUE_TIMEOUT_SECONDS = float(os.environ.get('LONG_QUEUE_TIMEOUT_SECONDS', '120'))
