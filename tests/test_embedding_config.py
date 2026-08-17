from app.embedding import config as c


def test_embedding_defaults():
    assert c.MAX_INPUT_TOKENS == 4096
    assert c.OPENVINO_DEVICE == 'GPU'
