from collections import defaultdict
from typing import Optional

from datasets import load_dataset as load_hf_dataset
from pydantic import BaseModel, ConfigDict, field_serializer

from event_dataset.common_schema import AbstractEvent
from event_dataset.entity import Entity
from event_dataset.schema import string_to_event_object
from log_utils import logger


class Example(BaseModel):
    id: str
    article: str
    gold_event_object: AbstractEvent
    predicted_event_type: Optional[str] = None
    predicted_event_object: Optional[AbstractEvent] = None
    linked_entities: Optional[list[Entity]] = None
    language: str

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",  # Disallow extra fields
    )

    @property
    def gold_event_type(self) -> str:
        return self.gold_event_object.get_event_type()

    @field_serializer("gold_event_object")
    def serialize_gold_event_object(self, gold_event_object: AbstractEvent) -> str:
        return repr(gold_event_object)

    @field_serializer("predicted_event_object")
    def serialize_predicted_event_object(
        self, predicted_event_object: AbstractEvent
    ) -> str | None:
        if predicted_event_object is None:
            return None
        return repr(predicted_event_object)


def load_dataset(
    dataset_id: str,
    data_split: str,
    language: Optional[str] = None,
    examples_per_language: int = -1,
) -> list[Example]:
    ret = []
    language_to_count = defaultdict(int)

    dataset = load_hf_dataset(dataset_id, split=data_split)
    for row in dataset:
        assert isinstance(row, dict)
        if language and row["language"] != language:
            continue

        if language_to_count[row["language"]] == examples_per_language:
            continue
        language_to_count[row["language"]] += 1

        if "linked_entities" in row:
            linked_entities = row["linked_entities"]
        else:
            linked_entities = None
        predicted_event_type = row.get("predicted_event_type", None)
        predicted_event_object = row.get("predicted_event_object", None)
        if predicted_event_object:
            predicted_event_object = string_to_event_object(predicted_event_object)

        example = Example(
            id=row["event_id_in_acled"],
            article=row["plain_text"],
            language=row["language"],
            gold_event_object=string_to_event_object(row["gold_label"]),
            predicted_event_type=predicted_event_type,
            predicted_event_object=predicted_event_object,
            linked_entities=linked_entities,
        )
        ret.append(example)

    logger.info(f"Loaded {len(ret):,} examples from {dataset_id} ({data_split})")
    return ret
