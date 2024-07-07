import pytest
from adapters.base_adapter import BaseAdapter
from plugins.llava_adapter import LLaVAAdapter
from plugins.japanese_stable_vlm_adapter import JapaneseStableVLMAdapter

@pytest.mark.asyncio
async def test_llava_adapter():
    adapter = LLaVAAdapter("liuhaotian/llava-v1.5-13b", "cpu", {"max_length": 50, "temperature": 0.7})
    assert adapter.model_name == "liuhaotian/llava-v1.5-13b"
    assert await adapter.verify()

    response = await adapter.generate_response("What's in this image?", "test.jpg")
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_japanese_stable_vlm_adapter():
    adapter = JapaneseStableVLMAdapter("stabilityai/japanese-stable-vlm", "cpu", {"max_length": 50, "temperature": 0.7})
    assert adapter.model_name == "stabilityai/japanese-stable-vlm"
    assert await adapter.verify()

    response = await adapter.generate_response("この画像には何が写っていますか？", "test.jpg")
    assert isinstance(response, str)
    assert len(response) > 0

def test_adapter_supports_model():
    assert LLaVAAdapter.supports_model("liuhaotian/llava-v1.5-13b")
    assert not LLaVAAdapter.supports_model("stabilityai/japanese-stable-vlm")
    assert JapaneseStableVLMAdapter.supports_model("stabilityai/japanese-stable-vlm")
    assert not JapaneseStableVLMAdapter.supports_model("liuhaotian/llava-v1.5-13b")

# Add more tests as needed