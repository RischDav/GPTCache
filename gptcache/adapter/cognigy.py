import base64
import json
import os
import time
from io import BytesIO
from typing import Any, AsyncGenerator, Iterator, List

from gptcache import cache
from gptcache.adapter.adapter import aadapt, adapt
from gptcache.adapter.base import BaseCacheLLM
from gptcache.manager.scalar_data.base import Answer, DataType
from gptcache.utils import import_pillow
from gptcache.utils.error import wrap_error
from gptcache.utils.response import (
    get_audio_text_from_openai_answer,
    get_image_from_openai_b64,
    get_image_from_openai_url,
    get_message_from_openai_answer,
    get_stream_message_from_openai_answer,
    get_text_from_openai_answer,
)
from gptcache.utils.token import token_counter


class Session(BaseCacheLLM):
    """Openai ChatCompletion Wrapper

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_prompt
            # init gptcache
            cache.init()
            cache.set_openai_key()

            from gptcache.adapter import openai
            # run ChatCompletion model with gptcache
            response = openai.ChatCompletion.create(
                          model='gpt-3.5-turbo',
                          messages=[
                            {
                                'role': 'user',
                                'content': "what's github"
                            }],
                        )
            response_content = response['choices'][0]['message']['content']
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        try:
            return (
                super().create(*llm_args, **llm_kwargs)
                if cls.llm is None
                else cls.llm(*llm_args, **llm_kwargs)
            )
        except openai.OpenAIError as e:
            raise wrap_error(e) from e

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs):
        try:
            return (
                (await super().acreate(*llm_args, **llm_kwargs))
                if cls.llm is None
                else await cls.llm(*llm_args, **llm_kwargs)
            )
        except openai.OpenAIError as e:
            raise wrap_error(e) from e

    @staticmethod
    def _update_cache_callback(
        llm_data, update_cache_func, *args, **kwargs
    ):  # pylint: disable=unused-argument
        if isinstance(llm_data, AsyncGenerator):

            async def hook_openai_data(it):
                total_answer = ""
                async for item in it:
                    total_answer += get_stream_message_from_openai_answer(item)
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook_openai_data(llm_data)
        elif not isinstance(llm_data, Iterator):
            update_cache_func(
                Answer(get_message_from_openai_answer(llm_data), DataType.STR)
            )
            return llm_data
        else:

            def hook_openai_data(it):
                total_answer = ""
                for item in it:
                    total_answer += get_stream_message_from_openai_answer(item)
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook_openai_data(llm_data)

    @classmethod
    def create(cls, *args, **kwargs):
        chat_cache = kwargs.get("cache_obj", cache)
        enable_token_counter = chat_cache.config.enable_token_counter

        def cache_data_convert(cache_data):
            if enable_token_counter:
                input_token = _num_tokens_from_messages(kwargs.get("messages"))
                output_token = token_counter(cache_data)
                saved_token = [input_token, output_token]
            else:
                saved_token = [0, 0]
            if kwargs.get("stream", False):
                return _construct_stream_resp_from_cache(cache_data, saved_token)
            return _construct_resp_from_cache(cache_data, saved_token)

        kwargs = cls.fill_base_args(**kwargs)
        return adapt(
            cls._llm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

    @classmethod
    async def acreate(cls, *args, **kwargs):
        chat_cache = kwargs.get("cache_obj", cache)
        enable_token_counter = chat_cache.config.enable_token_counter

        def cache_data_convert(cache_data):
            if enable_token_counter:
                input_token = _num_tokens_from_messages(kwargs.get("messages"))
                output_token = token_counter(cache_data)
                saved_token = [input_token, output_token]
            else:
                saved_token = [0, 0]
            if kwargs.get("stream", False):
                return async_iter(
                    _construct_stream_resp_from_cache(cache_data, saved_token)
                )
            return _construct_resp_from_cache(cache_data, saved_token)

        kwargs = cls.fill_base_args(**kwargs)
        return await aadapt(
            cls._allm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )


async def async_iter(input_list):
    for item in input_list:
        yield item


class Completion(openai.Completion, BaseCacheLLM):
    """Openai Completion Wrapper

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_prompt
            # init gptcache
            cache.init()
            cache.set_openai_key()

            from gptcache.adapter import openai
            # run Completion model with gptcache
            response = openai.Completion.create(model="text-davinci-003",
                                                prompt="Hello world.")
            response_text = response["choices"][0]["text"]
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        try:
            return (
                super().create(*llm_args, **llm_kwargs)
                if not cls.llm
                else cls.llm(*llm_args, **llm_kwargs)
            )
        except openai.OpenAIError as e:
            raise wrap_error(e) from e

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs):
        try:
            return (
                (await super().acreate(*llm_args, **llm_kwargs))
                if cls.llm is None
                else await cls.llm(*llm_args, **llm_kwargs)
            )
        except openai.OpenAIError as e:
            raise wrap_error(e) from e

    @staticmethod
    def _cache_data_convert(cache_data):
        return _construct_text_from_cache(cache_data)

    @staticmethod
    def _update_cache_callback(
        llm_data, update_cache_func, *args, **kwargs
    ):  # pylint: disable=unused-argument
        update_cache_func(Answer(get_text_from_openai_answer(llm_data), DataType.STR))
        return llm_data

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs = cls.fill_base_args(**kwargs)
        return adapt(
            cls._llm_handler,
            cls._cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

    @classmethod
    async def acreate(cls, *args, **kwargs):
        kwargs = cls.fill_base_args(**kwargs)
        return await aadapt(
            cls._allm_handler,
            cls._cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

def _construct_resp_from_cache(return_message, saved_token):
    return {
        "gptcache": True,
        "saved_token": saved_token,
        "choices": [
            {
                "message": {"role": "assistant", "content": return_message},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        "object": "chat.completion",
    }


def _construct_stream_resp_from_cache(return_message, saved_token):
    created = int(time.time())
    return [
        {
            "choices": [
                {"delta": {"role": "assistant"}, "finish_reason": None, "index": 0}
            ],
            "created": created,
            "object": "chat.completion.chunk",
        },
        {
            "choices": [
                {
                    "delta": {"content": return_message},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "created": created,
            "object": "chat.completion.chunk",
        },
        {
            "gptcache": True,
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "created": created,
            "object": "chat.completion.chunk",
            "saved_token": saved_token,
        },
    ]


def _construct_text_from_cache(return_text):
    return {
        "gptcache": True,
        "choices": [
            {
                "text": return_text,
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        "object": "text_completion",
    }


def _construct_image_create_resp_from_cache(image_data, response_format, size):
    import_pillow()
    from PIL import Image as PILImage  # pylint: disable=C0415

    img_bytes = base64.b64decode((image_data))
    img_file = BytesIO(img_bytes)  # convert image to file-like object
    img = PILImage.open(img_file)
    new_size = tuple(int(a) for a in size.split("x"))
    if new_size != img.size:
        img = img.resize(new_size)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
    else:
        buffered = img_file

    if response_format == "url":
        target_url = os.path.abspath(str(int(time.time())) + ".jpeg")
        with open(target_url, "wb") as f:
            f.write(buffered.getvalue())
        image_data = target_url
    elif response_format == "b64_json":
        image_data = base64.b64encode(buffered.getvalue()).decode("ascii")
    else:
        raise AttributeError(
            f"Invalid response_format: {response_format} is not one of ['url', 'b64_json']"
        )

    return {
        "gptcache": True,
        "created": int(time.time()),
        "data": [{response_format: image_data}],
    }


def _construct_audio_text_from_cache(return_text):
    return {
        "gptcache": True,
        "text": return_text,
    }


def _num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages."""
    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += token_counter(value)
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
