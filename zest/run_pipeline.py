import argparse
import asyncio
import json
from typing import Optional

import orjsonl
from chainlite import (
    get_all_configured_engines,
    get_total_cost,
    write_prompt_logs_to_file,
)

from event_dataset.entity import Entity, load_entity_database
from event_dataset.evaluate import FullEvaluationProcessor
from event_dataset.example import Example, load_dataset
from event_dataset.schema import Schema, load_from_module
from log_utils import logger
from zest.argument_extraction import ArgumentExtractor
from zest.entity_assignment import (
    EntityAssignmentProcessor,
)
from zest.entity_linking import EntityLinker
from zest.event_detection import EventTypeDetector
from zest.pipeline import ProcessingPipeline


class PipelineRunner:
    """Modular pipeline runner for event processing tasks."""

    def __init__(
        self,
        engine: str,
    ):
        self.engine = engine

    async def load_data(
        self,
        dataset_id: str,
        data_split: str,
        language: Optional[str] = None,
        examples_per_language: int = -1,
    ) -> tuple[list[Example], Schema]:
        """Load dataset and schema."""
        examples = load_dataset(
            dataset_id=dataset_id,
            data_split=data_split,
            language=language,
            examples_per_language=examples_per_language,
        )
        schema = load_from_module()
        return examples, schema

    async def setup_event_detection(
        self,
    ) -> EventTypeDetector:
        """Setup event detection components."""
        detector = EventTypeDetector(engine=self.engine)
        return detector

    async def setup_argument_extraction(self, schema: Schema) -> ArgumentExtractor:
        """Setup argument extraction components."""
        extractor = ArgumentExtractor(engine=self.engine)
        await extractor.setup(schema)
        return extractor

    async def setup_entity_linking(
        self, schema: Schema, entity_database_path: str
    ) -> EntityLinker:
        """Setup entity linking components."""
        entity_database: dict[str, Entity] = load_entity_database(entity_database_path)
        entity_linker = EntityLinker(
            engine=self.engine, entity_database=entity_database
        )
        await entity_linker.setup(schema)
        return entity_linker

    async def setup_entity_assignment(
        self, schema: Schema
    ) -> EntityAssignmentProcessor:
        """Setup entity assignment components."""
        processor = EntityAssignmentProcessor(engine=self.engine)
        await processor.setup(schema)
        return processor

    async def setup_full_evaluation(
        self, schema: Schema, entity_database_path: str, dataset_id: str
    ) -> FullEvaluationProcessor:
        """Setup full evaluation processor."""
        processor = FullEvaluationProcessor(
            engine=self.engine,
            entity_database_path=entity_database_path,
            dataset_id=dataset_id,
        )
        await processor.setup(schema)
        return processor

    async def run_full_pipeline(
        self,
        dataset_id: str,
        data_split: str,
        output_file: str,
        include_event_detection: bool,
        include_argument_extraction: bool,
        include_entity_linking: bool,
        include_entity_assignment: bool,
        language: Optional[str] = None,
        examples_per_language: int = -1,
        entity_database_path: Optional[str] = None,
    ) -> None:
        """Run configurable full pipeline with selected components."""
        logger.info(f"Starting full pipeline with engine: {self.engine}")

        # Load data
        examples, schema = await self.load_data(
            dataset_id, data_split, language, examples_per_language
        )

        processors = []

        # Setup event detection
        if include_event_detection:
            detector = await self.setup_event_detection()
            processors.append(detector)

        # Setup argument extraction
        if include_argument_extraction:
            extractor = await self.setup_argument_extraction(schema)
            processors.append(extractor)

        # Setup entity linking
        entity_linker = None
        if include_entity_linking and entity_database_path:
            entity_linker = await self.setup_entity_linking(
                schema, entity_database_path
            )
            processors.append(entity_linker)

        # Setup entity assignment
        if include_entity_assignment:
            entity_assignment_processor = await self.setup_entity_assignment(schema)
            processors.append(entity_assignment_processor)

        # Add full evaluation processor if needed
        if dataset_id and entity_database_path:
            logger.info("Setting up detailed evaluation processor...")
            eval_processor = await self.setup_full_evaluation(
                schema, entity_database_path, dataset_id
            )
            processors.append(eval_processor)

        # Create and run pipeline
        pipeline = ProcessingPipeline(processors=processors)

        try:
            processed_examples = await pipeline.run(examples, schema)

            # Save results
            orjsonl.save(
                output_file, [example.model_dump() for example in processed_examples]
            )

        finally:
            # Cleanup resources
            if entity_linker:
                await entity_linker.cleanup()


async def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run event processing pipeline")
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="stanford-oval/Lemonade",
        help="Dataset ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        required=True,
        choices=["dev", "test"],
        help="The data split to process.",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output jsonl file"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        required=True,
        help="The LLM to use",
    )
    parser.add_argument(
        "--language",
        default=None,
        type=str,
        help="Language code to evaluate on",
        choices=[
            "en",
            "es",
            "ar",
            "fr",
            "it",
            "ru",
            "de",
            "tr",
            "my",
            "id",
            "uk",
            "ko",
            "pt",
            "nl",
            "so",
            "ne",
            "zh",
            "fa",
            "he",
            "ja",
        ],
    )
    parser.add_argument(
        "--examples_per_language",
        type=int,
        default=-1,
        help="Number of input samples per language to process. Use -1 to process all samples. Subsampling is done from the end of each language data.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="full",
        choices=["event_detection", "full"],
        help="Pipeline task: 'event_detection' for event detection only, 'full' for all stages (event detection + argument extraction + entity linking + entity assignment)",
    )
    parser.add_argument(
        "--entity_database_path",
        type=str,
        default="event_dataset/entity_database.jsonl",
        help="Path to the entity JSONL file (enables entity linking when provided)",
    )

    args = parser.parse_args()
    logger.info(f"Input arguments: {json.dumps(vars(args), indent=2)}")

    all_configured_engines = get_all_configured_engines()
    if not all_configured_engines:
        raise ValueError(
            "No LLMs configured. Please set up llm_config.yaml and your API key before running the pipeline."
        )
    if not args.engine or args.engine not in all_configured_engines:
        raise ValueError(
            f"Invalid engine specified: {args.engine}. LLMs you have configured: {', '.join(all_configured_engines)}"
        )

    # Create pipeline runner
    runner = PipelineRunner(engine=args.engine)

    # Validate entity file requirement for entity linking
    if args.task == "full" and not args.entity_database_path:
        parser.error(
            "Entity file is required for full pipeline with entity linking. Please provide --entity_database_path."
        )

    # Define task configurations
    task_configs = {
        "event_detection": {
            "include_event_detection": True,
            "include_argument_extraction": False,
            "include_entity_linking": False,
            "include_entity_assignment": False,
        },
        "full": {
            "include_event_detection": True,
            "include_argument_extraction": True,
            "include_entity_linking": True,
            "include_entity_assignment": True,
        },
    }

    # Get configuration for the specified task
    config = task_configs.get(args.task)
    if not config:
        raise ValueError(f"Unknown task: {args.task}")

    # Run pipeline with the selected configuration
    await runner.run_full_pipeline(
        dataset_id=args.dataset_id,
        data_split=args.data_split,
        output_file=args.output_file,
        language=args.language,
        examples_per_language=args.examples_per_language,
        entity_database_path=args.entity_database_path,
        **config,
    )

    write_prompt_logs_to_file()  # for debugging LLM inputs and outputs
    logger.info(f"Pipeline completed successfully. Results saved to {args.output_file}")
    logger.info(f"Total LLM cost: ${get_total_cost():.3f}")


if __name__ == "__main__":
    asyncio.run(main())
