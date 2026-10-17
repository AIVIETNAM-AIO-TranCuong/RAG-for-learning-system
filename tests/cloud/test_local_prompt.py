"""Exercise local backend adapters without loading models or calling providers."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import src.llm as llm


def test_local_gemini_uses_separate_model_for_custom_instruction(monkeypatch):
    backend = object.__new__(llm.GeminiBackend)
    backend.model = Mock()
    backend.model.generate_content.return_value.text = 'default'
    custom = Mock()
    custom.generate_content.return_value.text = 'custom'
    factory = Mock(return_value=custom)
    monkeypatch.setitem(sys.modules, 'google.generativeai', SimpleNamespace(GenerativeModel=factory))
    monkeypatch.setattr(llm, 'settings', SimpleNamespace(gemini_model='test-model'))
    assert backend.generate('question', system_instruction='Tutor') == 'custom'
    assert backend.generate('question') == 'default'
    factory.assert_called_once_with('test-model', system_instruction='Tutor')


def test_local_completion_backends_receive_custom_instruction():
    hf = object.__new__(llm.HFLocalBackend)
    hf.pipe = Mock(return_value=[{'generated_text': 'OK'}])
    assert hf.generate('Question', system_instruction='Tutor') == 'OK'
    assert hf.pipe.call_args.args[0] == 'System instructions:\nTutor\n\nQuestion'
    hf.generate('Question')
    assert hf.pipe.call_args.args[0] == 'Question'

    vllm = object.__new__(llm.vLLMBackend)
    vllm.base_url = 'https://test.invalid'
    vllm.http_client = Mock()
    vllm.http_client.post.return_value.json.return_value = {'choices': [{'text': 'OK'}]}
    assert vllm.generate('Question', system_instruction='Tutor') == 'OK'
    assert vllm.http_client.post.call_args.kwargs['json']['prompt'] == 'System instructions:\nTutor\n\nQuestion'
