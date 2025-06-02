import json
from collections import defaultdict
from typing import Any, Optional

import json_repair
from chainlite import llm_generation_chain, run_async_in_parallel
from pydantic import BaseModel, Field
from tqdm import tqdm

from event_dataset.example import Example
from event_dataset.schema import Schema
from log_utils import logger
from zest.pipeline import BaseProcessor


class EntityAssignmentProcessor(BaseProcessor):
    """Processor for assigning linked entities to specific event argument fields."""

    def __init__(self, engine: str):
        super().__init__(engine)
        self.entity_assignment_chains = {}

    async def setup(self, schema: Schema) -> None:
        """Setup the entity assignment processor."""
        self._setup_chains(schema)

    def _setup_chains(self, schema: Schema):
        """Initialize the LLM chains for entity assignment."""
        for event_type in schema.concrete_event_names:
            entity_field_names_descriptions = (
                schema.get_entity_field_names_and_descriptions(event_type)
            )
            possible_fields_string = {
                "\n".join(
                    f"- {f}: {d}" for f, d in entity_field_names_descriptions.items()
                )
            }

            self.entity_assignment_chains[event_type] = llm_generation_chain(
                "assign_entities_to_arguments.prompt",
                engine=self.engine,
                max_tokens=2000,
                output_json=True,
                bind_prompt_values={
                    "possible_fields_string": possible_fields_string,
                    "event_type": event_type,
                    "schema": schema,
                },
            )

    async def process_batch(
        self, examples: list[Example], schema: Schema
    ) -> list[Example]:
        """Process a batch of examples to assign entities to event arguments."""

        entity_assignment_chain_inputs = defaultdict(list)

        for example in examples:
            assert (
                example.predicted_event_object is not None
            ), "predicted_event_object must be set in the example"
            event_without_entities, _ = (
                example.predicted_event_object.copy_with_entities_set_to_empty()
            )
            entity_assignment_chain_inputs[
                example.predicted_event_object.get_event_type()
            ].append(
                (
                    example.predicted_event_object,
                    {
                        "article": example.article,
                        "event_with_empty_entities": event_without_entities.model_dump_json(
                            indent=2
                        ),
                        "linked_entities": example.linked_entities,
                    },
                )
            )

        for event_type in tqdm(
            entity_assignment_chain_inputs, desc="Assigning Entities"
        ):
            predicted_event_objects, chain_inputs = zip(
                *entity_assignment_chain_inputs[event_type]
            )
            entity_assignment_dicts = await run_async_in_parallel(
                self.entity_assignment_chains[event_type].ainvoke,
                chain_inputs,
                max_concurrency=5,
            )
            assert len(entity_assignment_dicts) == len(predicted_event_objects)
            for entity_assignment, pred_event_object in zip(
                entity_assignment_dicts, predicted_event_objects
            ):
                assert (
                    pred_event_object.get_event_type() == event_type
                ), f"{pred_event_object.get_event_type()} != {event_type}"
                entity_assignment: dict[str, str] = json_repair.loads(entity_assignment)  # type: ignore
                if not isinstance(entity_assignment, dict):
                    logger.warning(
                        f"Entity assignment is not a dict: {entity_assignment}, type: {type(entity_assignment)}"
                    )
                    continue
                logger.debug(
                    f"entity assignment: {json.dumps(entity_assignment, indent=2, ensure_ascii=False)}"
                )

                # assign to the event object
                for field_name, entity in entity_assignment.items():
                    field_name = field_name.strip()
                    if field_name.startswith("-"):
                        field_name = field_name[1:].strip()
                    if not isinstance(entity, list):
                        logger.warning(f"Entity is not a list: {entity}")
                        entity = []
                    else:
                        # fix cases where LLM outputs [[actor1], [actor2]] instead of [actor1, actor2]
                        if len(entity) == 1 and isinstance(entity[0], list):
                            entity = entity[0]
                    if not isinstance(entity, list):
                        logger.warning(f"Entity is not a list: {entity}")
                        continue
                    entity = [e for e in entity if isinstance(e, str) and e]
                    # check if the field is in pred_event_object
                    if not hasattr(pred_event_object, field_name):
                        logger.warning(
                            f"Field {field_name} not found in event object, skipping assignment"
                        )
                    else:
                        setattr(pred_event_object, field_name, entity)
                # logger.info(f"pred_event_object: {pred_event_object}")

        return examples


class EntitySpan(BaseModel):
    """
    A span of text in the article that matches a given entity.
    """

    entity_name: str = Field(
        ...,
        description="The entity name, copied from the list of given entities",
    )
    explanation: Optional[str] = Field(
        ...,
        description="Explanation of why this span matches the given entity",
    )
    span: Optional[str] = Field(
        ...,
        description="The exact text from the article that mentions the entity",
    )


class AllEntitySpans(BaseModel):
    spans: list[EntitySpan]
