import os

MODEL_ID = os.environ.get('MODEL_ID', 'guoxuter/ov_intent_analysis_sft:v7_q8')
MODEL_DIR = os.environ.get('MODEL_DIR', '/models/OpenVINO/ov_intent_analysis_sft_int8_ov')
OPENVINO_DEVICE = os.environ.get('OPENVINO_DEVICE', 'GPU')
MAX_INPUT_TOKENS = int(os.environ.get('MAX_INPUT_TOKENS', '2048'))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', '128'))
TEMPERATURE = float(os.environ.get('TEMPERATURE', '0.0'))
