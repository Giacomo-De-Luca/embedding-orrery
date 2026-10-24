"""Gemma chat-template formatting (no model load).

Gemma's chat template opens the model turn with ``<start_of_turn>model\\n``.
Every generation path of ``GemmaPytorchInference`` must end its prompt with
that newline, or the first sampled token is spent on it (and steering runs on
a sequence the directions were not extracted from).
"""

from unittest.mock import MagicMock

from PIL import Image

from interpret.inference.gemma_pytorch import GemmaPytorchInference

GENERATION_PROMPT = "<start_of_turn>model\n"


def _wrapper_with_fake_model() -> GemmaPytorchInference:
    """A wrapper whose forked model is a mock, so nothing is loaded."""
    wrapper = object.__new__(GemmaPytorchInference)
    wrapper.device = "cpu"
    wrapper.model = MagicMock()
    wrapper.model.generate.return_value = ["unused"]
    wrapper.model.generate_stream.return_value = iter([])
    return wrapper


def _sequence_sent(model_call) -> list:
    """The interleaved prompt sequence a mocked generate call received."""
    first_arg = model_call.call_args.args[0]
    # generate takes a batch of sequences, generate_stream a single one.
    return first_arg[0] if isinstance(first_arg[0], list) else first_arg


def test_format_prompt_ends_with_generation_prompt():
    assert GemmaPytorchInference.format_prompt("Hi") == (
        "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n"
    )


def test_template_constants_match_the_formatted_turns():
    formatted = GemmaPytorchInference.format_prompt("Hi")
    assert formatted.startswith(GemmaPytorchInference.USER_TURN)
    assert formatted.endswith(GemmaPytorchInference.GENERATION_PROMPT)
    assert GemmaPytorchInference.GENERATION_PROMPT == GENERATION_PROMPT


def test_format_chat_single_user_turn_matches_format_prompt():
    assert GemmaPytorchInference.format_chat(
        [("user", "Hi")]
    ) == GemmaPytorchInference.format_prompt("Hi")


def test_format_chat_multi_turn_ends_with_generation_prompt():
    formatted = GemmaPytorchInference.format_chat(
        [("user", "What colour is the sky?"), ("model", "Blue."), ("user", "And grass?")]
    )
    assert formatted == (
        "<start_of_turn>user\nWhat colour is the sky?<end_of_turn>\n"
        "<start_of_turn>model\nBlue.<end_of_turn>\n"
        "<start_of_turn>user\nAnd grass?<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def test_format_chat_trailing_model_turn_stays_open_for_prefill():
    formatted = GemmaPytorchInference.format_chat([("user", "Count"), ("model", "One, two,")])
    assert formatted == ("<start_of_turn>user\nCount<end_of_turn>\n<start_of_turn>model\nOne, two,")


def test_generate_sends_the_newline_terminated_template():
    wrapper = _wrapper_with_fake_model()
    wrapper.generate("Hi", output_len=1)
    assert _sequence_sent(wrapper.model.generate) == [GemmaPytorchInference.format_prompt("Hi")]


def test_generate_stream_sends_the_newline_terminated_template():
    wrapper = _wrapper_with_fake_model()
    list(wrapper.generate_stream("Hi", output_len=1))
    assert _sequence_sent(wrapper.model.generate_stream) == [
        GemmaPytorchInference.format_prompt("Hi")
    ]


def test_generate_chat_sends_the_newline_terminated_template():
    wrapper = _wrapper_with_fake_model()
    turns = [("user", "Hi"), ("model", "Hello."), ("user", "Bye")]
    wrapper.generate_chat(turns, output_len=1)
    sequence = _sequence_sent(wrapper.model.generate)
    assert sequence == [GemmaPytorchInference.format_chat(turns)]
    assert sequence[0].endswith(GENERATION_PROMPT)


def test_generate_chat_continues_a_trailing_model_turn():
    wrapper = _wrapper_with_fake_model()
    wrapper.generate_chat([("user", "Count"), ("model", "One, two,")], output_len=1)
    assert _sequence_sent(wrapper.model.generate)[0].endswith(f"{GENERATION_PROMPT}One, two,")


def test_generate_chat_stream_sends_the_newline_terminated_template():
    wrapper = _wrapper_with_fake_model()
    turns = [("user", "Hi"), ("model", "Hello."), ("user", "Bye")]
    list(wrapper.generate_chat_stream(turns, output_len=1))
    sequence = _sequence_sent(wrapper.model.generate_stream)
    assert sequence == [GemmaPytorchInference.format_chat(turns)]
    assert sequence[0].endswith(GENERATION_PROMPT)


def test_generate_with_image_puts_the_image_in_the_user_turn(tmp_path):
    image_path = tmp_path / "pixel.png"
    Image.new("RGB", (1, 1)).save(image_path)
    wrapper = _wrapper_with_fake_model()
    wrapper.generate_with_image("Describe", str(image_path), output_len=1)

    before, image, after = _sequence_sent(wrapper.model.generate)
    assert before == "<start_of_turn>user\n"
    assert isinstance(image, Image.Image)
    assert after == f"Describe<end_of_turn>\n{GENERATION_PROMPT}"


def test_strip_prompt_drops_the_template_and_newline():
    decoded = GemmaPytorchInference.format_prompt("Hi") + "Hello there"
    assert GemmaPytorchInference._strip_prompt(decoded) == "Hello there"
