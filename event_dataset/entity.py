from typing import Optional

import orjsonl
from pydantic import BaseModel

from log_utils import logger


class Entity(BaseModel):
    name: str
    description: str

    def __hash__(self) -> int:
        return self.name.__hash__()

    @staticmethod
    def find_in_entity_db(
        entity_name: str, entity_db: dict[str, "Entity"]
    ) -> Optional["Entity"]:
        """If entity_name doesn't exist, finds the match that starts with entity_name"""
        if entity_name in entity_db:
            return entity_db[entity_name]
        for entity in entity_db:
            if entity.startswith(entity_name):
                logger.debug(
                    f"Found similar entity '{entity}' for '{entity_name}' in the entity database"
                )
                return entity_db[entity]
        logger.debug(f"Entity '{entity_name}' not found in the entity database")
        return None


def load_entity_database(entity_file: str) -> dict[str, Entity]:
    all_entity_descriptions: list[dict[str, str]] = orjsonl.load(entity_file)  # type: ignore
    entity_dict: dict[str, Entity] = {}
    for entity in all_entity_descriptions:
        entity_dict[entity["entity_name"]] = Entity(
            name=entity["entity_name"], description=entity["description"]
        )

    logger.info(f"Loaded {len(entity_dict):,} entities from {entity_file}")

    return entity_dict
