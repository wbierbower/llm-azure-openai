"""

"""

from llm import AsyncKeyModel, EmbeddingModel, KeyModel, hookimpl
import llm
from llm.utils import (
    # dicts_to_table_string,
    remove_dict_none_values,
    logging_client,
    simplify_usage_dict,
)
# import click
# import datetime
from enum import Enum
# import httpx
import openai
import os
from azure.identity import EnvironmentCredential, get_bearer_token_provider

from pydantic import field_validator, Field

from typing import AsyncGenerator, List, Iterable, Iterator, Optional, Union
import json
import yaml


@hookimpl
def register_models(register):
    # Load Azure OpenAI models from configuration
    azure_path = llm.user_dir() / "azure-openai-models.yaml"
    if not azure_path.exists():
        return
    with open(azure_path) as f:
        azure_models = yaml.safe_load(f)
    for azure_model in azure_models:
        model_id = azure_model["model_id"]
        aliases = azure_model.get("aliases", [])
        model_name = azure_model.get("model_name", model_id)
        api_key = azure_model.get("api_key")
        api_version = azure_model.get("api_version", "2023-12-01-preview")
        azure_endpoint = azure_model.get("azure_endpoint")
        kwargs = {}

        vision = azure_model.get("vision", False)
        audio = azure_model.get("audio", False)
        reasoning = azure_model.get("reasoning", False)
        supports_schema = azure_model.get("supports_schema", False)
        can_stream = azure_model.get("can_stream", True)
        allows_system_prompt = azure_model.get("allows_system_prompt", True)
        use_azure_ad = azure_model.get("use_azure_ad", True)

        # Set up Azure AD token provider if specified
        azure_ad_token_provider = None
        if use_azure_ad:
            azure_ad_token_provider = get_bearer_token_provider(
                EnvironmentCredential(), "https://cognitiveservices.azure.com/.default"
            )

        if can_stream is False:
            kwargs["can_stream"] = False

        if azure_model.get("completion"):
            klass = AzureCompletion
        else:
            klass = AzureChat

        azure_model_instance = klass(
            model_id,
            model_name=model_name,
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            vision=vision,
            audio=audio,
            reasoning=reasoning,
            supports_schema=supports_schema,
            allows_system_prompt=allows_system_prompt,
            azure_ad_token_provider=azure_ad_token_provider,
            **kwargs,
        )

        register(
            azure_model_instance,
            AsyncAzureChat(
                model_id,
                model_name=model_name,
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                vision=vision,
                audio=audio,
                reasoning=reasoning,
                supports_schema=supports_schema,
                allows_system_prompt=allows_system_prompt,
                azure_ad_token_provider=azure_ad_token_provider,
                **kwargs,
            ) if not azure_model.get("completion") else None,
            aliases=aliases,
        )


# @hookimpl
# def register_embedding_models(register):
#     # Load Azure OpenAI embedding models from configuration
#     azure_path = llm.user_dir() / "azure-openai-embeddings.yaml"
#     if not azure_path.exists():
#         return
#     with open(azure_path) as f:
#         azure_embeddings = yaml.safe_load(f)
#     for azure_embedding in azure_embeddings:
#         model_id = azure_embedding["model_id"]
#         aliases = azure_embedding.get("aliases", [])
#         model_name = azure_embedding.get('model_name', model_id)
#         api_key = azure_embedding.get("api_key")
#         api_version = azure_embedding.get("api_version", "2023-12-01-preview")
#         azure_endpoint = azure_embedding.get("azure_endpoint")
#         dimensions = azure_embedding.get("dimensions")
#         use_azure_ad = azure_embedding.get("use_azure_ad", False)

#         # Set up Azure AD token provider if specified
#         azure_ad_token_provider = None
#         if use_azure_ad:
#             azure_ad_token_provider = get_bearer_token_provider(
#                 EnvironmentCredential(), "https://cognitiveservices.azure.com/.default"
#             )

#         register(
#             AzureOpenAIEmbeddingModel(
#                 model_id,
#                 model_name,
#                 api_key,
#                 api_version,
#                 azure_endpoint,
#                 dimensions,
#                 azure_ad_token_provider
#             ),
#             aliases=aliases,
#         )


# @hookimpl
# def register_commands(cli):
#     @cli.group(name="azure-openai")
#     def azure_openai_():
#         "Commands for working directly with the Azure OpenAI API"

#     @azure_openai_.command()
#     @click.option("json_", "--json", is_flag=True, help="Output as JSON")
#     @click.option("--key", help="Azure OpenAI API key")
#     @click.option("--endpoint", help="Azure OpenAI endpoint")
#     @click.option("--api-version", default="2023-12-01-preview", help="API version")
#     def deployments(json_, key, endpoint, api_version):
#         "List deployments available to you from the Azure OpenAI API"
#         from llm import get_key

#         api_key = get_key(key, "azure-openai", "AZURE_OPENAI_API_KEY")
#         if not endpoint:
#             endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
#             if not endpoint:
#                 raise click.ClickException(
#                     "Azure OpenAI endpoint not provided. Please specify with --endpoint"
#                 )

#         response = httpx.get(
#             f"{endpoint}/openai/deployments?api-version={api_version}",
#             headers={
#                 "api-key": api_key,
#                 "Content-Type": "application/json"
#             },
#         )
#         if response.status_code != 200:
#             raise click.ClickException(
#                 f"Error {response.status_code} from Azure OpenAI API: {response.text}"
#             )
#         deployments = response.json()["data"]
#         if json_:
#             click.echo(json.dumps(deployments, indent=4))
#         else:
#             to_print = []
#             for deployment in deployments:
#                 created_str = deployment.get("created", "")
#                 if created_str and not isinstance(created_str, str):
#                     created_str = datetime.datetime.fromtimestamp(
#                         created_str, datetime.timezone.utc
#                     ).isoformat()
#                 to_print.append(
#                     {
#                         "id": deployment["id"],
#                         "model": deployment["model"],
#                         "status": deployment.get("status", ""),
#                         "created": created_str,
#                     }
#                 )
#             done = dicts_to_table_string("id model status created".split(), to_print)
#             print("\n".join(done))


class SharedOptions(llm.Options):
    temperature: Optional[float] = Field(
        description=(
            "What sampling temperature to use, between 0 and 2. Higher values like "
            "0.8 will make the output more random, while lower values like 0.2 will "
            "make it more focused and deterministic."
        ),
        ge=0,
        le=2,
        default=None,
    )
    max_tokens: Optional[int] = Field(
        description="Maximum number of tokens to generate.", default=None
    )
    top_p: Optional[float] = Field(
        description=(
            "An alternative to sampling with temperature, called nucleus sampling, "
            "where the model considers the results of the tokens with top_p "
            "probability mass. So 0.1 means only the tokens comprising the top "
            "10% probability mass are considered. Recommended to use top_p or "
            "temperature but not both."
        ),
        ge=0,
        le=1,
        default=None,
    )
    frequency_penalty: Optional[float] = Field(
        description=(
            "Number between -2.0 and 2.0. Positive values penalize new tokens based "
            "on their existing frequency in the text so far, decreasing the model's "
            "likelihood to repeat the same line verbatim."
        ),
        ge=-2,
        le=2,
        default=None,
    )
    presence_penalty: Optional[float] = Field(
        description=(
            "Number between -2.0 and 2.0. Positive values penalize new tokens based "
            "on whether they appear in the text so far, increasing the model's "
            "likelihood to talk about new topics."
        ),
        ge=-2,
        le=2,
        default=None,
    )
    stop: Optional[str] = Field(
        description=("A string where the API will stop generating further tokens."),
        default=None,
    )
    logit_bias: Optional[Union[dict, str]] = Field(
        description=(
            "Modify the likelihood of specified tokens appearing in the completion. "
            'Pass a JSON string like \'{"1712":-100, "892":-100, "1489":-100}\''
        ),
        default=None,
    )
    seed: Optional[int] = Field(
        description="Integer seed to attempt to sample deterministically",
        default=None,
    )

    @field_validator("logit_bias")
    def validate_logit_bias(cls, logit_bias):
        if logit_bias is None:
            return None

        if isinstance(logit_bias, str):
            try:
                logit_bias = json.loads(logit_bias)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in logit_bias string")

        validated_logit_bias = {}
        for key, value in logit_bias.items():
            try:
                int_key = int(key)
                int_value = int(value)
                if -100 <= int_value <= 100:
                    validated_logit_bias[int_key] = int_value
                else:
                    raise ValueError("Value must be between -100 and 100")
            except ValueError:
                raise ValueError("Invalid key-value pair in logit_bias dictionary")

        return validated_logit_bias


class ReasoningEffortEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class OptionsForReasoning(SharedOptions):
    json_object: Optional[bool] = Field(
        description="Output a valid JSON object {...}. Prompt must mention JSON.",
        default=None,
    )
    reasoning_effort: Optional[ReasoningEffortEnum] = Field(
        description=(
            "Constraints effort on reasoning for reasoning models. Currently supported "
            "values are low, medium, and high. Reducing reasoning effort can result in "
            "faster responses and fewer tokens used on reasoning in a response."
        ),
        default=None,
    )


def _attachment(attachment):
    url = attachment.url
    base64_content = ""
    if not url or attachment.resolve_type().startswith("audio/"):
        base64_content = attachment.base64_content()
        url = f"data:{attachment.resolve_type()};base64,{base64_content}"
    if attachment.resolve_type().startswith("image/"):
        return {"type": "image_url", "image_url": {"url": url}}
    else:
        format_ = "wav" if attachment.resolve_type() == "audio/wav" else "mp3"
        return {
            "type": "input_audio",
            "input_audio": {
                "data": base64_content,
                "format": format_,
            },
        }


class _AzureShared:
    def __init__(
        self,
        model_id,
        model_name=None,
        api_key=None,
        api_version="2023-12-01-preview",
        azure_endpoint=None,
        deployment=None,
        can_stream=True,
        vision=False,
        audio=False,
        reasoning=False,
        supports_schema=False,
        allows_system_prompt=True,
        azure_ad_token_provider=None,
    ):
        self.model_id = model_id
        self.model_name = model_name or model_id
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.can_stream = can_stream
        self.vision = vision
        self.supports_schema = supports_schema
        self.allows_system_prompt = allows_system_prompt
        self.azure_ad_token_provider = azure_ad_token_provider

        self.attachment_types = set()

        if reasoning:
            self.Options = OptionsForReasoning

        if vision:
            self.attachment_types.update(
                {
                    "image/png",
                    "image/jpeg",
                    "image/webp",
                    "image/gif",
                }
            )

        if audio:
            self.attachment_types.update(
                {
                    "audio/wav",
                    "audio/mpeg",
                }
            )

    def __str__(self):
        return "Azure OpenAI Chat: {}".format(self.model_id)

    def build_messages(self, prompt, conversation):
        messages = []
        current_system = None
        if conversation is not None:
            for prev_response in conversation.responses:
                if (
                    prev_response.prompt.system
                    and prev_response.prompt.system != current_system
                ):
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                    current_system = prev_response.prompt.system
                if prev_response.attachments:
                    attachment_message = []
                    if prev_response.prompt.prompt:
                        attachment_message.append(
                            {"type": "text", "text": prev_response.prompt.prompt}
                        )
                    for attachment in prev_response.attachments:
                        attachment_message.append(_attachment(attachment))
                    messages.append({"role": "user", "content": attachment_message})
                else:
                    messages.append(
                        {"role": "user", "content": prev_response.prompt.prompt}
                    )
                messages.append(
                    {"role": "assistant", "content": prev_response.text_or_raise()}
                )
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        if not prompt.attachments:
            messages.append({"role": "user", "content": prompt.prompt or ""})
        else:
            attachment_message = []
            if prompt.prompt:
                attachment_message.append({"type": "text", "text": prompt.prompt})
            for attachment in prompt.attachments:
                attachment_message.append(_attachment(attachment))
            messages.append({"role": "user", "content": attachment_message})
        return messages

    def set_usage(self, response, usage):
        if not usage:
            return
        input_tokens = usage.pop("prompt_tokens")
        output_tokens = usage.pop("completion_tokens")
        usage.pop("total_tokens")
        response.set_usage(
            input=input_tokens, output=output_tokens, details=simplify_usage_dict(usage)
        )

    def get_client(self, key, *, async_=False):
        kwargs = {}

        # Set Azure OpenAI specific properties
        kwargs["azure_endpoint"] = self.azure_endpoint
        kwargs["api_version"] = self.api_version

        # Use Azure AD token provider if available, otherwise fall back to API key
        if self.azure_ad_token_provider:
            # Set the token provider
            kwargs["azure_ad_token_provider"] = self.azure_ad_token_provider
        else:
            # Use Azure OpenAI key from initialization or get from key parameter
            if self.api_key:
                kwargs["api_key"] = self.api_key
            # elif self.needs_key:
            #     kwargs["api_key"] = self.get_key(key)
            else:
                # OpenAI-compatible models don't need a key, but the
                # openai client library requires one
                kwargs["api_key"] = "DUMMY_KEY"

        if os.environ.get("LLM_OPENAI_SHOW_RESPONSES"):
            kwargs["http_client"] = logging_client()

        if async_:
            model = openai.AsyncAzureOpenAI(**kwargs)
            return model
        else:
            model = openai.AzureOpenAI(**kwargs)
            return model

    def build_kwargs(self, prompt, stream):
        kwargs = dict(not_nulls(prompt.options))
        json_object = kwargs.pop("json_object", None)
        if "max_tokens" not in kwargs and self.default_max_tokens is not None:
            kwargs["max_tokens"] = self.default_max_tokens
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}
        if prompt.schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": prompt.schema},
            }
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs


class AzureChat(_AzureShared, KeyModel):
    # needs_key = "azure-openai"
    # key_env_var = "AZURE_OPENAI_API_KEY"
    default_max_tokens = None

    class Options(SharedOptions):
        json_object: Optional[bool] = Field(
            description="Output a valid JSON object {...}. Prompt must mention JSON.",
            default=None,
        )

    def execute(self, prompt, stream, response, conversation=None, key=None):
        if prompt.system and not self.allows_system_prompt:
            raise NotImplementedError("Model does not support system prompts")
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key)
        usage = None
        if stream:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                **kwargs,
            )
            chunks = []
            for chunk in completion:
                chunks.append(chunk)
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                try:
                    content = chunk.choices[0].delta.content
                except IndexError:
                    content = None
                if content is not None:
                    yield content
            response.response_json = remove_dict_none_values(combine_chunks(chunks))
        else:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False,
                **kwargs,
            )
            usage = completion.usage.model_dump()
            response.response_json = remove_dict_none_values(completion.model_dump())
            yield completion.choices[0].message.content
        self.set_usage(response, usage)
        response._prompt_json = redact_data({"messages": messages})


class AsyncAzureChat(_AzureShared, AsyncKeyModel):
    # needs_key = "azure-openai"
    # key_env_var = "AZURE_OPENAI_API_KEY"
    default_max_tokens = None

    class Options(SharedOptions):
        json_object: Optional[bool] = Field(
            description="Output a valid JSON object {...}. Prompt must mention JSON.",
            default=None,
        )

    async def execute(
        self, prompt, stream, response, conversation=None, key=None
    ) -> AsyncGenerator[str, None]:
        if prompt.system and not self.allows_system_prompt:
            raise NotImplementedError("Model does not support system prompts")
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key, async_=True)
        usage = None
        if stream:
            completion = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                **kwargs,
            )
            chunks = []
            async for chunk in completion:
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                chunks.append(chunk)
                try:
                    content = chunk.choices[0].delta.content
                except IndexError:
                    content = None
                if content is not None:
                    yield content
            response.response_json = remove_dict_none_values(combine_chunks(chunks))
        else:
            completion = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False,
                **kwargs,
            )
            response.response_json = remove_dict_none_values(completion.model_dump())
            usage = completion.usage.model_dump()
            yield completion.choices[0].message.content
        self.set_usage(response, usage)
        response._prompt_json = redact_data({"messages": messages})


class AzureCompletion(AzureChat):
    class Options(SharedOptions):
        logprobs: Optional[int] = Field(
            description="Include the log probabilities of most likely N per token",
            default=None,
            le=5,
        )

    def __init__(self, *args, default_max_tokens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_max_tokens = default_max_tokens

    def __str__(self):
        return "Azure OpenAI Completion: {}".format(self.model_id)

    def execute(self, prompt, stream, response, conversation=None, key=None):
        if prompt.system:
            raise NotImplementedError(
                "System prompts are not supported for Azure OpenAI completion models"
            )
        messages = []
        if conversation is not None:
            for prev_response in conversation.responses:
                messages.append(prev_response.prompt.prompt)
                messages.append(prev_response.text())
        messages.append(prompt.prompt)
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key)
        if stream:
            completion = client.completions.create(
                model=self.model_name,
                prompt="\n".join(messages),
                stream=True,
                **kwargs,
            )
            chunks = []
            for chunk in completion:
                chunks.append(chunk)
                try:
                    content = chunk.choices[0].text
                except IndexError:
                    content = None
                if content is not None:
                    yield content
            combined = combine_chunks(chunks)
            cleaned = remove_dict_none_values(combined)
            response.response_json = cleaned
        else:
            completion = client.completions.create(
                model=self.model_name,
                prompt="\n".join(messages),
                stream=False,
                **kwargs,
            )
            response.response_json = remove_dict_none_values(completion.model_dump())
            yield completion.choices[0].text
        response._prompt_json = redact_data({"messages": messages})


class AzureOpenAIEmbeddingModel(EmbeddingModel):
    # needs_key = "azure-openai"
    # key_env_var = "AZURE_OPENAI_API_KEY"
    batch_size = 100

    def __init__(self, model_id, model_name, api_key=None, api_version="2023-12-01-preview", azure_endpoint=None, dimensions=None, azure_ad_token_provider=None):
        self.model_id = model_id
        self.model_name = model_name
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.dimensions = dimensions
        self.azure_ad_token_provider = azure_ad_token_provider

    def embed_batch(self, items: Iterable[Union[str, bytes]]) -> Iterator[List[float]]:
        kwargs = {
            "input": items,
            "model": self.model_name,
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        azure_kwargs = {}
        # Set Azure OpenAI specific properties
        azure_kwargs["azure_endpoint"] = self.azure_endpoint
        azure_kwargs["api_version"] = self.api_version

        # Use Azure AD token provider if available, otherwise fall back to API key
        if self.azure_ad_token_provider:
            azure_kwargs["azure_ad_token_provider"] = self.azure_ad_token_provider
        else:
            if self.api_key:
                azure_kwargs["api_key"] = self.api_key
            else:
                azure_kwargs["api_key"] = self.get_key()

        client = openai.AzureOpenAI(**azure_kwargs)
        results = client.embeddings.create(**kwargs).data
        return ([float(r) for r in result.embedding] for result in results)


def not_nulls(data) -> dict:
    return {key: value for key, value in data if value is not None}


def combine_chunks(chunks: List) -> dict:
    content = ""
    role = None
    finish_reason = None
    # If any of them have log probability, we're going to persist
    # those later on
    logprobs = []
    usage = {}

    for item in chunks:
        if item.usage:
            usage = item.usage.dict()
        for choice in item.choices:
            if choice.logprobs and hasattr(choice.logprobs, "top_logprobs"):
                logprobs.append(
                    {
                        "text": choice.text if hasattr(choice, "text") else None,
                        "top_logprobs": choice.logprobs.top_logprobs,
                    }
                )

            if not hasattr(choice, "delta"):
                content += choice.text
                continue
            role = choice.delta.role
            if choice.delta.content is not None:
                content += choice.delta.content
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

    # Imitations of the OpenAI API may be missing some of these fields
    combined = {
        "content": content,
        "role": role,
        "finish_reason": finish_reason,
        "usage": usage,
    }
    if logprobs:
        combined["logprobs"] = logprobs
    if chunks:
        for key in ("id", "object", "model", "created", "index"):
            value = getattr(chunks[0], key, None)
            if value is not None:
                combined[key] = value

    return combined


def redact_data(input_dict):
    """
    Recursively search through the input dictionary for any 'image_url' keys
    and modify the 'url' value to be just 'data:...'.

    Also redact input_audio.data keys
    """
    if isinstance(input_dict, dict):
        for key, value in input_dict.items():
            if (
                key == "image_url"
                and isinstance(value, dict)
                and "url" in value
                and value["url"].startswith("data:")
            ):
                value["url"] = "data:..."
            elif key == "input_audio" and isinstance(value, dict) and "data" in value:
                value["data"] = "..."
            else:
                redact_data(value)
    elif isinstance(input_dict, list):
        for item in input_dict:
            redact_data(item)
    return input_dict
