"""
Event type detection module for the processing pipeline.
"""

import textwrap
from functools import partial
from typing import Any

from chainlite import llm_generation_chain, run_async_in_parallel
from chainlite.llm_output import extract_tag_from_llm_output

from event_dataset.example import Example
from event_dataset.schema import Schema
from log_utils import logger
from zest.pipeline import BaseProcessor


class EventTypeDetector(BaseProcessor):
    """Event type detection processor."""

    def __init__(self, engine: str):
        super().__init__(engine)
        self._setup_chain()

    def _setup_chain(self):
        """Setup the LLM chain for event detection."""
        input_str = """First, here is the news article you need to analyze:
            <article>
            {{ article }}
            </article>

            Now, carefully review the annotation guidelines for various event types:

            {% for event_name in schema.concrete_event_names %}
            [{{ loop.index }}] "{{ event_name }}": {{ schema.event_name_to_description(event_name) }}

            {% endfor %}


            1. For each event type, determine how well it matches the main event in the article. Consider the following factors.

            2. Output your result in this format:

            """
        input_str += (
            "<answer>\nevent type, as one word. E.g. PeacefulProtest\n</answer>"
        )

        template_blocks = [
            (
                "instruction",
                """You will be provided with annotation guidelines and a news article to analyze. Your goal is to identify the most relevant event type for the main event described in the news article. The main event is the event the writer of the article is writing the article for. This excludes events that are provided to familiarize the audience with the background of the news story.""",
            ),
            (
                "input",
                textwrap.dedent(input_str),
            ),
        ]

        self.chain = llm_generation_chain(
            "detect_event_type",
            template_blocks=template_blocks,
            engine=self.engine,
            max_tokens=200,
            stop_tokens=["</answer>"],
        )

        def extract_answer(result):
            """Extract answer from LLM output."""
            return extract_tag_from_llm_output.invoke(result, tags="answer")  # type: ignore

        self.chain = self.chain | extract_answer

    async def _predict_single_event_type(self, example: Example, schema: Schema) -> str:
        """Predict event type for a single example."""
        event_type: str = await self.chain.ainvoke(
            {"article": example.article, "schema": schema}
        )
        return event_type

    async def process_batch(
        self, examples: list[Example], schema: Schema
    ) -> list[Example]:
        """Process a batch of examples to predict event types."""
        predicted_event_types = await run_async_in_parallel(
            partial(self._predict_single_event_type, schema=schema),
            examples,
            max_concurrency=100,
            desc="Predicting Event Types",
        )

        # Clean up predictions and handle invalid event types
        predicted_event_types = self._clean_predictions(predicted_event_types, schema)

        # Update examples with predictions
        for example, prediction in zip(examples, predicted_event_types):
            example.predicted_event_type = prediction

        return examples

    def _clean_predictions(self, predictions: list[str], schema: Schema) -> list[str]:
        """Clean and validate predictions against schema."""
        cleaned_predictions = []

        for event_type in predictions:
            if event_type not in schema.concrete_event_names:
                if ":" in event_type:
                    event_type = event_type.split(":")[-1].strip()
                event_type_clean = event_type.lower().strip()
                best_event = min(
                    schema.concrete_event_names,
                    key=lambda possible: levenshtein_distance(
                        possible.lower().strip(), event_type_clean
                    ),
                )
                logger.warning(
                    f"Predicted event type {event_type} is not in the schema. Replacing with {best_event}."
                )
                event_type = best_event

            cleaned_predictions.append(event_type)

        return cleaned_predictions


def f1_score(y_true: list[str], y_pred: list[str], average: str = "micro") -> float:
    """Simple F1 score implementation for micro averaging."""
    if average == "micro":
        # For multi-class, micro F1 equals accuracy
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true) if y_true else 0.0
    else:
        raise NotImplementedError("Only micro averaging is implemented")


def levenshtein_distance(a: str, b: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(a) < len(b):
        return levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
