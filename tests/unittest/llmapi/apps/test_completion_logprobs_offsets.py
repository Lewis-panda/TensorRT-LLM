# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Completion logprobs locate token starts in the returned text."""

import json

import pytest
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from tensorrt_llm.executor.result import CompletionOutput, GenerationResultBase, Logprob
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.serve.postprocess_handlers import (
    CompletionPostprocArgs,
    completion_response_post_processor,
    completion_stream_post_processor,
    create_completion_logprobs,
)

pytestmark = pytest.mark.cpu_only


def _tokenizer(tokens: list[str]) -> TransformersTokenizer:
    backend = Tokenizer(models.WordLevel(dict(zip(tokens, range(len(tokens))))))
    backend.decoder = decoders.Fuse()
    return TransformersTokenizer(PreTrainedTokenizerFast(tokenizer_object=backend))


@pytest.mark.parametrize("top_logprobs", [False, True])
@pytest.mark.parametrize(
    ("tokens", "initial_offset", "expected"),
    [
        (["A", " wonderful", " world", "!"], 0, [0, 1, 11, 17]),
        (["é", "中文", "!"], 4, [4, 5, 7]),
        (["first", "", "last"], 0, [0, 5, 5]),
        (["only"], 9, [9]),
        ([], 7, []),
    ],
)
def test_logprob_offsets_identify_token_starts(
    tokens: list[str], initial_offset: int, expected: list[int], top_logprobs: bool
) -> None:
    token_ids = list(range(len(tokens)))
    logprobs = (
        [{token_id: Logprob(logprob=-0.5)} for token_id in token_ids]
        if top_logprobs
        else [-0.5] * len(tokens)
    )
    result = create_completion_logprobs(token_ids, _tokenizer(tokens), logprobs, initial_offset)

    assert result.text_offset == expected
    assert result.tokens == tokens
    assert result.token_logprobs == [-0.5] * len(tokens)
    assert result.top_logprobs == ([{token: -0.5} for token in tokens] if top_logprobs else [])


@pytest.mark.parametrize("streaming", [False, True])
def test_completion_response_logprobs_point_into_choice_text(streaming: bool) -> None:
    tokens = ["é", " wonderful", " world", "!"]
    tokenizer = _tokenizer(tokens)
    result = GenerationResultBase(1, SamplingParams(max_tokens=4))
    output = CompletionOutput(
        index=0,
        text=tokenizer.decode([0, 1, 2, 3]),
        token_ids=[0, 1, 2, 3],
        logprobs=[-0.5] * 4,
        finish_reason="length",
    )
    result._outputs = [output]
    result._done = True
    args = CompletionPostprocArgs(
        model="test", tokenizer=tokenizer, num_prompt_tokens=1, return_logprobs=True
    )

    if streaming:
        # The first token was delivered in an earlier chunk. The next chunk's
        # offsets remain relative to the full text, measured in characters.
        output._last_text_len = 1
        output._last_token_ids_len = 1
        output._last_logprobs_len = 1
        chunk = completion_stream_post_processor(result, args)[0]
        choice = json.loads(chunk.removeprefix("data: "))["choices"][0]
        assert choice["text"] == " wonderful world!"
        assert choice["logprobs"]["text_offset"] == [1, 11, 17]
    else:
        choice = completion_response_post_processor(result, args).choices[0].model_dump()
        assert choice["text"] == "é wonderful world!"
        assert choice["logprobs"]["text_offset"] == [0, 1, 11, 17]

    for token, offset in zip(choice["logprobs"]["tokens"], choice["logprobs"]["text_offset"]):
        assert output.text[offset : offset + len(token)] == token
