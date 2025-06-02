import abc
from typing import Any, Type, Union, get_args, get_origin

from pydantic import BaseModel


def empty_for_type(typ: Any) -> Any:
    origin = get_origin(typ)
    args = get_args(typ)
    if origin is Union:
        # If Optional[T] or union including None, pick T.
        non_none = [arg for arg in args if arg is not type(None)]
        if non_none:
            return empty_for_type(non_none[0])
        return None
    if typ is int:
        return 0
    if typ is float:
        return 0.0
    if typ is str:
        return ""
    if typ is bool:
        return False
    if origin is list:
        return []
    if origin is tuple:
        return ()
    if origin is dict:
        return {}
    # If the type defines its own empty(), call it.
    if hasattr(typ, "empty") and callable(getattr(typ, "empty")):
        return typ.empty()
    # Fallback: return None.
    return None


class AbstractEvent(BaseModel, abc.ABC):
    """A general class to represent events."""

    def __init__(self, **data: Any):
        super().__init__(**data)
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, list):
                field_value.sort()

    def get_event_type(self) -> str:
        return self.__class__.__name__

    def get_entity_field_names(self) -> list[str]:
        return [
            field_name
            for field_name, field_info in self.__class__.model_fields.items()
            if field_info.json_schema_extra
            and isinstance(field_info.json_schema_extra, dict)
            and field_info.json_schema_extra.get("is_entity_field") == True
        ]

    def get_entity_field_values(self) -> list[str]:
        """Find the value of all fields that are marked with is_entity_field=True"""
        entity_field_values = [
            getattr(self, field_name) for field_name in self.get_entity_field_names()
        ]
        # Flatten the list of lists
        return [entity for sublist in entity_field_values for entity in sublist]

    def copy_with_entities_set_to_empty(
        self,
    ) -> tuple["AbstractEvent", list[str]]:
        new_event = self.model_copy()
        entity_fields = []
        for field_name in self.get_entity_field_names():
            if hasattr(new_event, field_name):
                setattr(new_event, field_name, [])
                entity_fields.append(field_name)

        return new_event, entity_fields

    @classmethod
    def empty(cls: type["AbstractEvent"]) -> "AbstractEvent":
        """
        Returns an instance of the class with "empty" values for every field.
        """
        data: dict[str, Any] = {}
        for field_name, model_field in cls.model_fields.items():
            if not model_field.is_required:
                if model_field.default_factory is not None:
                    data[field_name] = model_field.default_factory
                else:
                    data[field_name] = model_field.default
            else:
                annotation = model_field.annotation or Any
                # Handle the case where annotation is Any type
                if annotation is Any:
                    data[field_name] = None
                else:
                    data[field_name] = empty_for_type(annotation)
        return cls.model_validate(data)

    def __eq__(self, other):
        if not isinstance(other, AbstractEvent):
            return NotImplemented
        # Compare all fields for equality, converting lists to tuples
        return all(
            (
                tuple(getattr(self, field_name))
                if isinstance(getattr(self, field_name), list)
                else getattr(self, field_name)
            )
            == (
                tuple(getattr(other, field_name))
                if isinstance(getattr(other, field_name), list)
                else getattr(other, field_name)
            )
            for field_name in self.model_fields
        )
