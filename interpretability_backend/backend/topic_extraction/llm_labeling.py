"""Provider-agnostic LLM labeling for topic extraction.

Supports Gemini (default) and OpenAI for generating human-readable topic labels
from clustered documents and c-TF-IDF keywords.
"""

import logging
import os
import time
from collections.abc import Callable

logger = logging.getLogger("star_map." + __name__)


DEFAULT_SYSTEM_PROMPT = "You are an assistant that extracts high-level topics from texts."

DEFAULT_TOPIC_PROMPT = """You will extract a short topic label from given documents and keywords.
Here are two examples of topics you created before:

# Example 1
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the worst food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat beef eat eating emissions steak food health processed chicken
topic: Environmental impacts of eating meat

# Example 2
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.
- I got a message stating that I received the monitor but that is not true!
- It took a month longer to deliver than was advised...

Keywords: deliver weeks product shipping long delivery received arrived arrive week
topic: Shipping and delivery issues

# Your task
Sample texts from this topic:
{documents}

Keywords: {keywords}

Based on the information above, extract a short topic label (three words at most) in the following format:
topic: <topic_label>
"""


class _GeminiLabeler:
    """Generate topic labels using Google Gemini."""

    def __init__(self, model: str = "gemini-3-flash-preview"):
        from google import genai
        from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=api_key)
        self.model = model

        self.generate_label = retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )(self._generate_label)

    def _generate_label(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model, contents=prompt, config={"system_instruction": DEFAULT_SYSTEM_PROMPT}
        )
        text = response.text
        if text is None:
            raise ValueError("Gemini returned no text")
        return text.strip()


class _OpenAILabeler:
    """Generate topic labels using OpenAI."""

    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

        api_key = os.environ.get("CHROMA_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("CHROMA_OPENAI_API_KEY not set")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

        self.generate_label = retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )(self._generate_label)

    def _generate_label(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stop="\n",
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()


def _create_labeler(provider: str, model: str):
    """Factory function to create the appropriate labeler.

    Args:
        provider: "gemini" or "openai"
        model: Model name for the provider

    Returns:
        A labeler instance with a generate_label(prompt) method
    """
    provider = provider.lower()
    if provider == "gemini":
        return _GeminiLabeler(model=model)
    elif provider == "openai":
        return _OpenAILabeler(model=model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'gemini' or 'openai'.")


def generate_llm_label_for_topic(
    topic_id: int,
    keywords: list[tuple[str, float]],
    sample_documents: list[str],
    labeler,
) -> str | None:
    """Generate an LLM label for a single topic.

    Args:
        topic_id: The topic ID being labeled
        keywords: List of (word, score) tuples for this topic
        sample_documents: Representative documents for this topic
        labeler: Pre-created labeler instance (from _create_labeler())

    Returns:
        Label string or None on failure.
    """
    keyword_str = ", ".join([w for w, _ in keywords[:10]])
    docs_str = "\n".join([f"- {doc[:200]}" for doc in sample_documents[:4]])

    prompt = DEFAULT_TOPIC_PROMPT.format(documents=docs_str, keywords=keyword_str)

    try:
        call_start = time.time()
        label = labeler.generate_label(prompt)
        call_duration = time.time() - call_start
        label = label.replace("topic: ", "").replace("Topic: ", "")
        logger.info(f'Topic {topic_id} -> "{label}" ({call_duration:.1f}s)')
        return label
    except Exception as e:
        call_duration = time.time() - call_start
        logger.warning(f"Topic {topic_id} FAILED after {call_duration:.1f}s: {e}")
        return None


def generate_llm_labels(
    topics_data: dict[int, list[tuple[str, float]]],
    documents_df,
    llm_provider: str = "gemini",
    llm_model: str = "gemini-3-flash-preview",
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[int, str]:
    """Generate human-readable topic labels using an LLM.

    Args:
        topics_data: Dict of topic_id -> list of (word, score) tuples
        documents_df: DataFrame with Document_ID, Document, Topic columns
        llm_provider: LLM provider to use ("gemini" or "openai")
        llm_model: Model name for the provider
        progress_callback: Optional callback(done, total) called after each topic is labeled

    Returns:
        Dict of topic_id -> label string
    """
    try:
        labeler = _create_labeler(llm_provider, llm_model)
    except ValueError as e:
        logger.warning(f"LLM labeling unavailable: {e}")
        return {}
    except ImportError as e:
        logger.warning(f"LLM package not installed for provider '{llm_provider}': {e}")
        return {}

    labels = {}
    topics_to_label = [tid for tid in topics_data.keys() if tid != -1]
    total_calls = len(topics_to_label)
    logger.info(f"LLM labeling: {total_calls} topics to label using {llm_provider}/{llm_model}")
    labeling_start = time.time()

    for call_num, topic_id in enumerate(topics_to_label, 1):
        keywords = topics_data[topic_id]
        topic_docs = documents_df[documents_df["Topic"] == topic_id]["Document"].tolist()

        label = generate_llm_label_for_topic(
            topic_id=topic_id,
            keywords=keywords,
            sample_documents=topic_docs,
            labeler=labeler,
        )
        if label is not None:
            labels[topic_id] = label
        logger.info(f"[{call_num}/{total_calls}] done")
        if progress_callback is not None:
            progress_callback(call_num, total_calls)

    total_duration = time.time() - labeling_start
    logger.info(
        f"LLM labeling complete: {len(labels)}/{total_calls} labeled in {total_duration:.1f}s"
    )
    return labels
