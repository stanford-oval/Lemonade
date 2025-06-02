from abc import ABC, abstractmethod

from event_dataset.example import Example
from event_dataset.schema import Schema
from log_utils import logger


class BaseProcessor(ABC):
    """Base class for all processing modules in the pipeline."""

    def __init__(self, engine: str):
        self.engine = engine
        self.chain = None

    @abstractmethod
    async def process_batch(
        self, examples: list[Example], schema: Schema
    ) -> list[Example]:
        """Process a batch of examples and return updated examples."""
        pass

    @property
    def name(self) -> str:
        """Return the name of this processor for logging."""
        return self.__class__.__name__


class ProcessingPipeline:
    """Main processing pipeline that can handle multiple modules."""

    def __init__(self, processors: list[BaseProcessor]):
        self.processors = processors

    async def run(self, examples: list[Example], schema: Schema) -> list[Example]:
        """Run the complete processing pipeline."""
        current_examples = examples

        # Process through each module
        for processor in self.processors:
            logger.info(f"Running {processor.name}...")
            current_examples = await processor.process_batch(current_examples, schema)

        return current_examples
