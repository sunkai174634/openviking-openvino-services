from app.intent import config as c


def test_intent_defaults():
    assert c.MODEL_ID == 'guoxuter/ov_intent_analysis_sft:v7_q8'
    assert c.TEMPERATURE == 0.1  # aligned with v7 prompt since f141964
