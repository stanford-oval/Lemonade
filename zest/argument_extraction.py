from typing import Any

from chainlite import llm_generation_chain, run_async_in_parallel

from event_dataset.example import Example
from event_dataset.schema import Schema
from log_utils import logger
from zest.pipeline import BaseProcessor


class ArgumentExtractor(BaseProcessor):
    """Processor for extracting event arguments."""

    def __init__(self, engine: str):
        super().__init__(engine)
        self.eae_chains: dict[str, Any] = {}
        self.empty_events: dict[str, Any] = {}

    async def setup(self, schema: Schema) -> None:
        """Setup the argument extractor with schema."""
        # Create empty events
        for event_type in schema.concrete_event_names:
            self.empty_events[event_type] = schema.event_name_to_class(
                event_type
            ).empty()

        # Create EAE chains
        for event_type in schema.concrete_event_names:
            self.eae_chains[event_type] = self._create_eae_chain(
                schema=schema,
                event_type=event_type,
            )

    def _create_eae_chain(self, schema: Schema, event_type: str):
        """Create EAE chain for a specific event type."""
        template_blocks = [
            (
                "instruction",
                f"""You are an AI # filepath: /home/sina/multi-news/public_release/run_pipeline.pystant tasked with extracting event arguments from a given news article. You will be provided with annotation guidelines for an event type and a news article to analyze.
                    Extract the arguments of the main event in the article, which is of type {event_type}.

                    {event_type}: {schema.event_name_to_description(event_type)}

                When extracting event arguments, only pay attention to the main event in the article. Do not include any background information or other previous events that may be mentioned in the article.
                """,
            ),
            (
                "input",
                """{{ article }}""",
            ),
        ]

        return llm_generation_chain(
            template_file=f"eae_{event_type}",
            template_blocks=template_blocks,
            engine=self.engine,
            max_tokens=1500,
            pydantic_class=schema.event_name_to_class(event_type),
        )

    async def _process(self, example: Example) -> Example:
        """Process a single example to extract event arguments."""
        event_type = example.predicted_event_type

        if not event_type or event_type not in self.eae_chains:
            example.predicted_event_object = None
            return example

        eae_outputs = await self.eae_chains[event_type].ainvoke(
            {"article": example.article}
        )
        if not eae_outputs:
            example.predicted_event_object = self.empty_events[event_type]
        else:
            example.predicted_event_object = eae_outputs

        return example

    async def process_batch(
        self, examples: list[Example], schema: Schema
    ) -> list[Example]:
        """Process a batch of examples to extract event arguments."""

        return await run_async_in_parallel(
            self._process,
            examples,
            max_concurrency=20,
            desc="Extracting event arguments",
        )
