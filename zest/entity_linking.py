from collections import defaultdict

import json_repair
from chainlite import chain, llm_generation_chain, run_async_in_parallel
from chainlite.llm_output import extract_tag_from_llm_output

from event_dataset.entity import Entity
from event_dataset.example import Example
from event_dataset.schema import Schema
from event_dataset.schema_definition import Event
from log_utils import logger
from zest.entity_assignment import AllEntitySpans
from zest.entity_retriever import EntityRetriever, QueryResult
from zest.pipeline import BaseProcessor


@chain
def process_entity_evidence_output(all_spans: AllEntitySpans) -> dict[str, set]:
    ret = {}
    if not all_spans or getattr(all_spans, "spans", None) is None:
        return ret
    for s in all_spans.spans:
        entity_name = s.entity_name
        if entity_name not in ret:
            ret[entity_name] = []
        if s.span:
            ret[entity_name].append(s.span)

    return ret


def json_to_values(x) -> list:
    """Convert JSON string to list of values."""
    try:
        parsed = json_repair.loads(x)
        if isinstance(parsed, dict):
            ret = list(parsed.values())
            ret = [a for a in ret if a and a.strip()]  # remove empty strings
            return ret
        else:
            return []
    except Exception:
        logger.debug(f"Error parsing JSON: {x}")
        return []


class EntityLinker(BaseProcessor):
    """Entity linking processor with proper async context management."""

    def __init__(self, engine: str, entity_database: dict[str, Entity]):
        super().__init__(engine)
        self.entity_database = entity_database
        self.retriever = EntityRetriever()
        self.entity_query_generation_chain = None
        self.entity_extraction_chain = None

    async def setup(self, schema: Schema) -> None:
        """Setup the entity linker."""
        await self.retriever.setup()
        self._setup_chains()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.retriever.cleanup()

    def _setup_chains(self):
        """Initialize the LLM chains."""
        self.entity_query_generation_chain = (
            llm_generation_chain(
                "generate_entity_queries.prompt",
                engine=self.engine,
                max_tokens=2000,
            )
            | extract_tag_from_llm_output.bind(tags="entity_list")  # type: ignore
            | json_to_values
        )

        self.entity_extraction_chain = (
            llm_generation_chain(
                "extract_entity_evidence.prompt",
                engine=self.engine,
                max_tokens=2000,
                pydantic_class=AllEntitySpans,
            )
            | process_entity_evidence_output
        )

    async def retrieve_entity_via_search_api(
        self,
        queries: list[str],
        num_results: int,
        batch_size: int,
        retriever_endpoint: str = "https://search.genie.stanford.edu/lemonade_entities",
    ) -> list[QueryResult]:
        """
        Retrieve search results from a retriever API using the EntityRetriever.
        """
        return await self.retriever.retrieve_entity_via_search_api(
            queries=queries,
            num_results=num_results,
            batch_size=batch_size,
            retriever_endpoint=retriever_endpoint,
        )

    def _get_example_country(self, example: Example) -> str:
        """Extract country from example's predicted event object."""
        if not example.predicted_event_object:
            return ""

        location = getattr(example.predicted_event_object, "location", None)
        if location:
            return getattr(location, "country", "")
        return ""

    def _build_query_input(self, example: Example) -> dict[str, str]:
        """Build query input for a single example."""
        return {
            "article": example.article,
            "country": (
                example.predicted_event_object.location.country
                if isinstance(example.predicted_event_object, Event)
                else ""
            ),
        }

    async def _generate_entity_queries(
        self, examples: list[Example]
    ) -> list[list[str]]:
        """Generate entity queries for a batch of examples."""
        assert (
            self.entity_query_generation_chain is not None
        ), "Entity query generation chain is not set up."

        query_inputs = [self._build_query_input(e) for e in examples]

        queries_grouped_by_article = await run_async_in_parallel(
            self.entity_query_generation_chain.ainvoke,
            query_inputs,
            max_concurrency=20,
            desc="Entity Query Generation",
        )
        logger.debug(f"Queries generated: {queries_grouped_by_article}")
        return queries_grouped_by_article

    async def _retrieve_and_group_entities(
        self, queries_grouped_by_article: list[list[str]], num_results_per_query: int
    ) -> list[list[QueryResult]]:
        """Retrieve entities and group them by article."""
        all_queries = [q for sublist in queries_grouped_by_article for q in sublist]

        flattened_retrieved_entities = await self.retrieve_entity_via_search_api(
            queries=all_queries,
            num_results=num_results_per_query,
            batch_size=20,
        )

        assert len(flattened_retrieved_entities) == len(all_queries), (
            f"Length mismatch between queries and retrieved entities: "
            f"{len(all_queries)} vs {len(flattened_retrieved_entities)}"
        )

        # Truncate each result to num_results_per_query
        for qr in flattened_retrieved_entities:
            qr.results = qr.results[:num_results_per_query]

        # Group results back by article
        entities_grouped_by_article = []
        query_index = 0
        for query_group in queries_grouped_by_article:
            entities_grouped_by_article.append(
                flattened_retrieved_entities[
                    query_index : query_index + len(query_group)
                ]
            )
            query_index += len(query_group)

        return entities_grouped_by_article

    def _convert_query_results_to_entities(
        self, entities_grouped_by_article: list[list[QueryResult]]
    ) -> list[list[Entity]]:
        """Convert QueryResult objects to Entity objects."""
        all_entities = []
        for article_result in entities_grouped_by_article:
            entities = []
            for qr in article_result:
                for result in qr.results:
                    entities.append(
                        Entity(name=result.document_title, description=result.content)
                    )

            # Deduplicate while maintaining consistent order
            entities = list(dict.fromkeys(entities))
            all_entities.append(entities)

        return all_entities

    def _create_entity_extraction_inputs(
        self, examples: list[Example], all_entities: list[list[Entity]], shard_size: int
    ) -> list[dict]:
        """Create flattened inputs for entity extraction with sharding."""
        flat_inputs = []

        for i, (example, entities) in enumerate(zip(examples, all_entities)):
            shards = [
                entities[j : j + shard_size]
                for j in range(0, len(entities), shard_size)
            ]
            for shard in shards:
                flat_inputs.append(
                    {
                        "article": example.article,
                        "entities": shard,
                        "country": self._get_example_country(example),
                        "example_index": i,
                    }
                )

        return flat_inputs

    async def _extract_entities_with_evidence(
        self, flat_inputs: list[dict]
    ) -> dict[int, list[Entity]]:
        """Extract entities that have evidence in the text."""
        flat_responses = await run_async_in_parallel(
            self.entity_extraction_chain.ainvoke,  # type: ignore
            flat_inputs,
            max_concurrency=50,
            desc="Extractive Entity Filtering",
        )

        grouped_results: dict[int, list[Entity]] = defaultdict(list)

        for input_item, response in zip(flat_inputs, flat_responses):
            if not response:
                continue
            idx = input_item["example_index"]
            for entity_str, has_evidence in response.items():
                if has_evidence:
                    entity_obj = Entity.find_in_entity_db(
                        entity_str, self.entity_database
                    )
                    if entity_obj:
                        grouped_results[idx].append(entity_obj)

        return grouped_results

    async def do_entity_linking_batch(
        self,
        examples: list[Example],
        num_results_per_query: int = 40,
        shard_size: int = 5,
    ) -> list[list[Entity]]:
        """Perform entity linking on a batch of examples."""
        # Generate queries
        queries_grouped_by_article = await self._generate_entity_queries(examples)

        # Retrieve and group entities
        entities_grouped_by_article = await self._retrieve_and_group_entities(
            queries_grouped_by_article, num_results_per_query
        )

        # Convert to Entity objects
        all_entities = self._convert_query_results_to_entities(
            entities_grouped_by_article
        )
        assert len(all_entities) == len(examples)

        # Create inputs for entity extraction
        flat_inputs = self._create_entity_extraction_inputs(
            examples, all_entities, shard_size
        )

        # Extract entities with evidence
        grouped_results = await self._extract_entities_with_evidence(flat_inputs)

        # Rebuild the original all_entities list (ordered by examples)
        final_entities = [grouped_results[i] for i in range(len(examples))]

        return final_entities

    def _augment_predicted_entities(
        self, predicted_entities: set[str], example: Example, schema: Schema
    ) -> set[str]:
        """Augment predicted entities based on event type and characteristics."""
        predicted_abstract_event_type = (
            schema.concrete_event_name_to_abstract_event_name(
                example.predicted_event_type
            )
            if example.predicted_event_type
            else ""
        )

        # TODO move to evaluation instead
        if predicted_abstract_event_type == "Protest":
            predicted_entities.add("Protestors")
        elif predicted_abstract_event_type == "Riot":
            predicted_entities.add("Rioters")

        if getattr(example.predicted_event_object, "targets_civilians", False):
            predicted_entities.add("Civilians")

        return predicted_entities

    def _convert_entity_names_to_objects(self, entity_names: set[str]) -> list[Entity]:
        """Convert entity names to Entity objects."""
        entities = []
        for entity_name in entity_names:
            entity_obj = Entity.find_in_entity_db(entity_name, self.entity_database)
            if entity_obj:
                entities.append(entity_obj)
        return entities

    def _calculate_metrics(
        self, examples: list[Example], all_predicted_entities: list[set[str]]
    ) -> tuple[int, int, int, int]:
        """Calculate precision/recall metrics."""
        tp = fp = fn = num_predictions = 0

        for example, predicted_entities in zip(examples, all_predicted_entities):
            gold_entities = set(example.gold_event_object.get_entity_field_values())

            tp += len(gold_entities & predicted_entities)
            fp += len(predicted_entities - gold_entities)
            fn += len(gold_entities - predicted_entities)
            num_predictions += len(predicted_entities)

            # Log missing predicted entities
            for entity in gold_entities - predicted_entities:
                logger.info(f"Missing entity: {entity}")

        return tp, fp, fn, num_predictions

    def _process_example_entities(
        self, example: Example, example_entities: list[Entity], schema: Schema
    ) -> set[str]:
        """Process entities for a single example."""
        # Deduplicate and convert to names
        predicted_entity_names = set(
            entity.name for entity in list(dict.fromkeys(example_entities))
        )

        # Augment with rule-based entities
        predicted_entity_names = self._augment_predicted_entities(
            predicted_entity_names, example, schema
        )

        # Convert back to Entity objects and assign to example
        example.linked_entities = self._convert_entity_names_to_objects(
            predicted_entity_names
        )

        return predicted_entity_names

    async def process_batch(
        self, examples: list[Example], schema: Schema
    ) -> list[Example]:
        """Process a batch of examples for entity linking."""
        linked_entities = await self.do_entity_linking_batch(
            examples=examples,
            num_results_per_query=40,
        )

        # Process entities for each example
        all_predicted_entities = [
            self._process_example_entities(example, example_entities, schema)
            for example, example_entities in zip(examples, linked_entities)
        ]

        # Calculate and log metrics
        tp, fp, fn, num_predictions = self._calculate_metrics(
            examples, all_predicted_entities
        )

        if tp + fn > 0:  # Avoid division by zero
            recall = tp / (tp + fn) * 100
            avg_predictions = num_predictions // len(examples)
            logger.info(f"Recall@{avg_predictions}: {recall:.1f}")

        return examples
