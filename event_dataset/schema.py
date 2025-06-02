import importlib.util
import os
import sys
from abc import ABC
from functools import lru_cache
from types import ModuleType

from pydantic import BaseModel, ConfigDict

from event_dataset.common_schema import AbstractEvent


class Schema(BaseModel):
    """Class representing the entire schema for a dataset like Lemonade."""

    concrete_event_classes: list[type[AbstractEvent]]
    abstract_event_classes: list[type[AbstractEvent]]

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",  # Disallow extra fields
    )

    @property
    def concrete_event_names(self) -> list[str]:
        return [cls.__name__ for cls in self.concrete_event_classes]

    def event_name_to_description(self, event_name: str) -> str:
        for cls in self.concrete_event_classes:
            if cls.__name__ == event_name:
                assert (
                    cls.__doc__ is not None
                ), f"Class {cls.__name__} does not have a docstring."
                return cls.__doc__.strip()
        raise ValueError(f"Event name {event_name} not found in schema")

    def event_name_to_class(self, event_name: str) -> type:
        for cls in self.concrete_event_classes:
            if cls.__name__ == event_name:
                return cls
        raise ValueError(f"Event name {event_name} not found in schema")

    def concrete_event_name_to_abstract_event_name(self, event_name: str) -> str:
        for cls in self.concrete_event_classes:
            if cls.__name__ == event_name:
                return cls.__bases__[0].__name__
        raise ValueError(f"Event name {event_name} not found in schema")

    def get_entity_field_names_and_descriptions(
        self, event_name: str
    ) -> dict[str, str]:
        entity_fields = {}
        for field_name, field_info in self.event_name_to_class(
            event_name
        ).model_fields.items():
            if (
                field_info.json_schema_extra
                and field_info.json_schema_extra.get("is_entity_field") == True
            ):
                entity_fields[field_name] = field_info.description

        return entity_fields


@lru_cache
def load_module_from_path(file_path, module_name="schema_definition") -> ModuleType:
    # Create a module spec from the file location
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot find module named {module_name} at {file_path}")

    if spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} - no loader available")

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules
    sys.modules[module_name] = module

    # Execute the module in its own namespace
    spec.loader.exec_module(module)

    return module


def get_schema_path() -> str:
    return os.path.join("event_dataset", "schema_definition.py")


def load_from_module() -> "Schema":
    """Load the dataset schema from the module path."""
    schema_definition: ModuleType = load_module_from_path(get_schema_path())

    # find all classes that are subclasses of BaseModel
    all_event_classes = [
        cls
        for cls in schema_definition.__dict__.values()
        if isinstance(cls, type)
        and issubclass(cls, AbstractEvent)
        and cls != AbstractEvent
    ]

    # find all event_classes that are defined as a direct subclass of abc.ABC
    abstract_event_classes = [cls for cls in all_event_classes if ABC in cls.__bases__]
    concrete_event_classes = [
        cls for cls in all_event_classes if cls not in abstract_event_classes
    ]

    return Schema(
        abstract_event_classes=abstract_event_classes,
        concrete_event_classes=concrete_event_classes,
    )


def string_to_event_object(string: str) -> AbstractEvent:
    if not isinstance(string, str):
        return string
    module = load_module_from_path(
        get_schema_path(),
        module_name="schema_definition",
    )
    module_dict = {
        attr: getattr(module, attr) for attr in dir(module) if not attr.startswith("__")
    }

    return eval(string.strip(), {"__builtins__": None}, module_dict)
