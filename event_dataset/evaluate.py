import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import orjsonl
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

from event_dataset.example import Example, load_dataset
from event_dataset.schema import Schema, string_to_event_object
from log_utils import logger
from zest.pipeline import BaseProcessor

# Constants
ENTITY_KEYS = [
    "entities",
    "seen_entities",
    "unseen_entities",
    "generic_entities",
    "specific_entities",
]

EVALUATION_CATEGORIES = [
    "event_type",
    "entities",
    "seen_entities",
    "unseen_entities",
    "generic_entities",
    "specific_entities",
    "non_entity_args",
]

DEFAULT_NUM_BUCKETS = 10


@dataclass
class MetricCounts:
    """Data class for tracking true positives, false positives, and false negatives."""

    tp: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class MetricScores:
    """Data class for computed precision, recall, and F1 scores."""

    tp: float
    fp: float
    fn: float
    precision: float
    recall: float
    f1: float


@dataclass
class EventBreakdown:
    """Data class for event breakdown information."""

    event_type: str = ""
    entities: list[str] = field(default_factory=list)
    seen_entities: list[str] = field(default_factory=list)
    unseen_entities: list[str] = field(default_factory=list)
    generic_entities: list[str] = field(default_factory=list)
    specific_entities: list[str] = field(default_factory=list)
    entity_args: list[str] = field(default_factory=list)
    non_entity_args: list[str] = field(default_factory=list)
    language: str = ""
    syntax_error: bool = False


class FrequencyBucketer:
    """Handles logarithmic bucketing of entities by frequency."""

    def __init__(
        self, frequencies: list[float], num_buckets: int = DEFAULT_NUM_BUCKETS
    ):
        self.bucket_thresholds = self._compute_bucket_thresholds(
            frequencies, num_buckets
        )

    def _compute_bucket_thresholds(
        self, frequencies: list[float], num_buckets: int
    ) -> dict[str, float]:
        """Compute logarithmic bucket thresholds."""
        nonzero_freqs = [f for f in frequencies if f > 0]
        if not nonzero_freqs:
            raise ValueError(
                "No nonzero frequencies available for logarithmic bucketing."
            )

        log_freqs = np.log10(nonzero_freqs)
        log_min, log_max = log_freqs.min(), log_freqs.max()
        log_thresholds = np.linspace(log_min, log_max, num_buckets + 1)[1:]
        bucket_labels = [f"p{10*(i+1)}" for i in range(num_buckets)]

        return {
            label: 10**log_threshold
            for label, log_threshold in zip(bucket_labels, log_thresholds)
        }

    def filter_entities_by_frequency(
        self,
        entities: list[str],
        entity_to_db_item: dict[str, dict[str, Any]],
        threshold: float,
    ) -> list[str]:
        """Filter entities by frequency threshold."""
        return [
            e
            for e in entities
            if e in entity_to_db_item and entity_to_db_item[e]["frequency"] <= threshold
        ]


class MetricsCalculator:
    """Handles calculation of evaluation metrics."""

    @staticmethod
    def compute_scores(counts: MetricCounts) -> MetricScores:
        """Compute precision, recall, and F1 from counts."""
        precision = (
            counts.tp / (counts.tp + counts.fp) if (counts.tp + counts.fp) > 0 else 0.0
        )
        recall = (
            counts.tp / (counts.tp + counts.fn) if (counts.tp + counts.fn) > 0 else 0.0
        )
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        return MetricScores(
            tp=float(counts.tp),
            fp=float(counts.fp),
            fn=float(counts.fn),
            precision=precision,
            recall=recall,
            f1=f1,
        )

    @staticmethod
    def update_list_metrics(
        gold: list[str], pred: list[str], counts: MetricCounts
    ) -> None:
        """Update metrics for list-based comparisons."""
        gold_set = set(gold)
        pred_set = set(pred)
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        counts.tp += tp
        counts.fp += fp
        counts.fn += fn

    @staticmethod
    def update_event_type_metrics(gold: str, pred: str, counts: MetricCounts) -> None:
        """Update metrics for event type comparison."""
        if gold == pred:
            counts.tp += 1
        else:
            counts.fp += 1
            counts.fn += 1


class MetricsFactory:
    """Factory class for creating metrics data structures."""

    @staticmethod
    def create_metrics_dict(bucket_thresholds: dict[str, float]) -> dict[str, Any]:
        """Create empty metrics dictionary."""
        return {
            "metrics": {cat: MetricCounts() for cat in EVALUATION_CATEGORIES},
            "buckets": {
                bucket: {key: MetricCounts() for key in ENTITY_KEYS}
                for bucket in bucket_thresholds
            },
        }

    @staticmethod
    def create_language_metrics() -> dict[str, dict[str, Any]]:
        """Create empty language metrics dictionary."""
        return {}


class MetricsAggregator:
    """Helper class to aggregate metrics and reduce code duplication."""

    def __init__(self, calculator: MetricsCalculator):
        self.calculator = calculator

    def create_aggregated_counts(
        self, entities_counts: MetricCounts, non_entity_counts: MetricCounts
    ) -> MetricCounts:
        """Create aggregated counts from entities and non_entity_args."""
        return MetricCounts(
            tp=entities_counts.tp + non_entity_counts.tp,
            fp=entities_counts.fp + non_entity_counts.fp,
            fn=entities_counts.fn + non_entity_counts.fn,
        )

    def compute_category_scores(
        self, metrics_data: dict[str, MetricCounts]
    ) -> dict[str, dict[str, Any]]:
        """Compute scores for all categories in metrics data."""
        return {
            cat: self.calculator.compute_scores(counts).__dict__
            for cat, counts in metrics_data.items()
        }

    def compute_bucket_scores(
        self, bucket_data: dict[str, dict[str, MetricCounts]]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Compute scores for all buckets."""
        return {
            bucket_label: {
                key: self.calculator.compute_scores(counts).__dict__
                for key, counts in bucket_counts.items()
            }
            for bucket_label, bucket_counts in bucket_data.items()
        }

    def add_aggregated_score(
        self, results: dict[str, Any], metrics_data: dict[str, MetricCounts]
    ) -> None:
        """Add aggregated entities+non_entity_args score to results."""
        entities_counts = metrics_data["entities"]
        non_entity_counts = metrics_data["non_entity_args"]
        agg_counts = self.create_aggregated_counts(entities_counts, non_entity_counts)
        results["entities+non_entity_args"] = self.calculator.compute_scores(
            agg_counts
        ).__dict__


class MetricsTracker:
    """Helper class to track and update metrics during evaluation."""

    def __init__(
        self,
        calculator: MetricsCalculator,
        bucketer: FrequencyBucketer,
        entity_to_db_item: dict[str, dict[str, Any]],
    ):
        self.calculator = calculator
        self.bucketer = bucketer
        self.entity_to_db_item = entity_to_db_item

    def update_category_metrics(
        self,
        gold_breakdown: EventBreakdown,
        pred_breakdown: EventBreakdown,
        metrics_dict: dict[str, MetricCounts],
    ) -> None:
        """Update metrics for all evaluation categories."""
        for category in EVALUATION_CATEGORIES:
            if category == "event_type":
                self.calculator.update_event_type_metrics(
                    gold_breakdown.event_type,
                    pred_breakdown.event_type,
                    metrics_dict[category],
                )
            else:
                gold_val = getattr(gold_breakdown, category, [])
                pred_val = getattr(pred_breakdown, category, [])
                self.calculator.update_list_metrics(
                    gold_val, pred_val, metrics_dict[category]
                )

    def update_bucket_metrics(
        self,
        gold_breakdown: EventBreakdown,
        pred_breakdown: EventBreakdown,
        bucket_metrics: dict[str, dict[str, MetricCounts]],
    ) -> None:
        """Update bucket metrics for all entity keys."""
        for bucket_label, threshold in self.bucketer.bucket_thresholds.items():
            for key in ENTITY_KEYS:
                gold_entities = getattr(gold_breakdown, key, [])
                pred_entities = getattr(pred_breakdown, key, [])

                gold_filtered = self.bucketer.filter_entities_by_frequency(
                    gold_entities, self.entity_to_db_item, threshold
                )
                pred_filtered = self.bucketer.filter_entities_by_frequency(
                    pred_entities, self.entity_to_db_item, threshold
                )

                self.calculator.update_list_metrics(
                    gold_filtered, pred_filtered, bucket_metrics[bucket_label][key]
                )

    def update_all_metrics(
        self,
        gold_breakdown: EventBreakdown,
        pred_breakdown: EventBreakdown,
        *metrics_targets: dict[str, Any],
    ) -> None:
        """Update both category and bucket metrics for multiple targets."""
        for metrics_target in metrics_targets:
            self.update_category_metrics(
                gold_breakdown, pred_breakdown, metrics_target["metrics"]
            )
            self.update_bucket_metrics(
                gold_breakdown, pred_breakdown, metrics_target["buckets"]
            )


class ScoreCalculationHelper:
    """Helper class for consistent score calculation across different metric types."""

    @staticmethod
    def calculate_metrics_dict_scores(
        metrics_data: dict[str, Any], aggregator: MetricsAggregator
    ) -> dict[str, Any]:
        """Calculate scores for a metrics data structure."""
        result = {}
        result.update(aggregator.compute_category_scores(metrics_data["metrics"]))
        aggregator.add_aggregated_score(result, metrics_data["metrics"])
        result["buckets"] = aggregator.compute_bucket_scores(metrics_data["buckets"])
        return result


class EventProcessor:

    def __init__(self, entity_to_db_item: dict[str, dict[str, Any]]):
        self.entity_to_db_item = entity_to_db_item

    def process_event(
        self, event_object: Any, *, split_kwargs: dict[str, Any]
    ) -> EventBreakdown:
        """Process event object to extract breakdown information."""
        breakdown = EventBreakdown()
        breakdown.event_type = event_object.get_event_type()

        # Get and annotate entities
        entities = list(set(event_object.get_entity_field_values()))
        entities = [e for e in entities if e in self.entity_to_db_item]

        breakdown.entities = entities
        breakdown.seen_entities = [
            e for e in entities if self.entity_to_db_item[e]["seen"]
        ]
        breakdown.unseen_entities = [
            e for e in entities if not self.entity_to_db_item[e]["seen"]
        ]
        breakdown.generic_entities = [
            e for e in entities if self.entity_to_db_item[e]["entity_type"] == "generic"
        ]
        breakdown.specific_entities = [
            e
            for e in entities
            if self.entity_to_db_item[e]["entity_type"] == "specific"
        ]

        # Split additional arguments
        location_args, entity_args, other_args = event_object.split_to_arguments(
            include_event_type=True, **split_kwargs
        )
        breakdown.entity_args = entity_args
        breakdown.non_entity_args = other_args + location_args

        return breakdown

    def break_down_example(
        self, example: Example
    ) -> tuple[EventBreakdown, EventBreakdown]:
        """Break down a single example into gold and predicted components."""
        gold_breakdown = self.process_event(example.gold_event_object, split_kwargs={})
        gold_breakdown.language = example.language
        gold_location = getattr(example.gold_event_object, "location", None)

        try:
            pred_breakdown = self.process_event(
                example.predicted_event_object,
                split_kwargs={"gold_location": gold_location},
            )
            pred_breakdown.syntax_error = False
        except Exception as e:
            logger.warning(f"Encountered syntax error: {e}")
            pred_breakdown = EventBreakdown(syntax_error=True)

        return gold_breakdown, pred_breakdown


class FullEvaluationProcessor(BaseProcessor):
    """Processor for comprehensive evaluation with detailed metrics breakdown."""

    def __init__(self, engine: str, entity_database_path: str, dataset_id: str):
        super().__init__(engine)
        self.entity_database_path = entity_database_path
        self.dataset_id = dataset_id
        self.entity_to_db_item: dict[str, dict[str, Any]] = {}
        self.formatter = MetricsFormatter()

    async def setup(self, schema: Schema) -> None:
        """Setup the evaluation processor with entity database and training data."""
        # Load entity database
        entity_database = orjsonl.load(self.entity_database_path)
        self.entity_to_db_item = {}
        for item in entity_database:
            if isinstance(item, dict) and "entity_name" in item:
                self.entity_to_db_item[item["entity_name"]] = item

        # Load seen entities from training set
        self._load_seen_entities()

    async def process_batch(
        self, examples: list[Example], schema: Schema
    ) -> list[Example]:
        """Process batch and generate comprehensive evaluation metrics."""
        # Run evaluation on the batch
        examples = [self._process(example, schema) for example in examples]
        metrics = self._aggregate_metrics_by_language(examples)

        # Also log JSON for programmatic access
        logger.info("Raw metrics (JSON):")
        logger.info(json.dumps(metrics, indent=2, ensure_ascii=False))

        # Display human-readable results
        self.formatter.format_metrics(metrics)
        # Display summary statistics
        self._display_summary_stats(metrics)

        return examples

    def _load_seen_entities(self) -> None:
        """Load and mark seen entities from training set."""
        train_set = load_dataset(dataset_id=self.dataset_id, data_split="train")

        logger.info("Extracting seen entities from the training set...")
        seen_entities = set()
        for row in train_set:
            if isinstance(row, dict) and "gold_label" in row:
                gold = string_to_event_object(row["gold_label"])
                seen_entities.update(gold.get_entity_field_values())

        for e, db_entry in self.entity_to_db_item.items():
            if e in seen_entities:
                db_entry["seen"] = True
            else:
                db_entry["seen"] = False

    def _process(self, example: Example, schema: Schema) -> Example:
        """Process a single example."""
        # If predicted event object is not set, set it to the empty class of type predicted_event_type
        if example.predicted_event_object is None:
            predicted_event_type = example.predicted_event_type
            assert (
                predicted_event_type is not None
            ), "Predicted event type must be set if predicted event object is None"
            predicted_event_class = schema.event_name_to_class(predicted_event_type)
            example.predicted_event_object = predicted_event_class.empty()
        return example

    def _aggregate_metrics_by_language(
        self, examples: list[Example]
    ) -> dict[str, dict[str, Any]]:
        """Aggregate evaluation metrics by language."""
        # Initialize frequency bucketer
        frequencies = [
            db_item["frequency"] for db_item in self.entity_to_db_item.values()
        ]
        bucketer = FrequencyBucketer(frequencies)

        # Initialize metrics tracking
        language_metrics = MetricsFactory.create_language_metrics()
        overall_metrics = MetricsFactory.create_metrics_dict(bucketer.bucket_thresholds)

        # Process all examples
        event_processor = EventProcessor(self.entity_to_db_item)
        for example in tqdm(
            examples, desc="Processing examples for evaluation", smoothing=0
        ):
            gold_breakdown, pred_breakdown = event_processor.break_down_example(example)
            lang = gold_breakdown.language or "unknown"

            if lang not in language_metrics:
                language_metrics[lang] = MetricsFactory.create_metrics_dict(
                    bucketer.bucket_thresholds
                )

            # Update metrics for this example
            self._update_metrics_for_example(
                gold_breakdown,
                pred_breakdown,
                language_metrics[lang],
                overall_metrics,
                bucketer,
            )

        # Compute final scores
        return self._compute_final_scores(
            language_metrics, overall_metrics, bucketer.bucket_thresholds
        )

    def _update_metrics_for_example(
        self,
        gold_breakdown: EventBreakdown,
        pred_breakdown: EventBreakdown,
        lang_metrics: dict[str, Any],
        overall_metrics: dict[str, Any],
        bucketer: FrequencyBucketer,
    ) -> None:
        """Update metrics for a single example."""
        calculator = MetricsCalculator()
        tracker = MetricsTracker(calculator, bucketer, self.entity_to_db_item)

        # Update all metrics for both language and overall tracking
        tracker.update_all_metrics(
            gold_breakdown, pred_breakdown, lang_metrics, overall_metrics
        )

    def _compute_final_scores(
        self,
        language_metrics: dict[str, dict[str, Any]],
        overall_metrics: dict[str, Any],
        bucket_thresholds: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
        """Compute final scores from collected metrics."""
        calculator = MetricsCalculator()
        aggregator = MetricsAggregator(calculator)
        results: dict[str, Any] = {}

        # Process per-language results
        for lang, data in language_metrics.items():
            results[lang] = ScoreCalculationHelper.calculate_metrics_dict_scores(
                data, aggregator
            )

        # Overall results
        results["all"] = ScoreCalculationHelper.calculate_metrics_dict_scores(
            overall_metrics, aggregator
        )

        # Include bucket thresholds
        results["bucket_thresholds"] = bucket_thresholds

        return results

    def _display_summary_stats(self, results: dict[str, Any]) -> None:
        """Display a quick summary of key statistics."""
        if "all" not in results:
            return

        overall = results["all"]
        console = Console()

        # Create summary panel
        summary_text = []

        if "event_type" in overall:
            event_type_f1 = overall["event_type"].get("f1", 0)
            summary_text.append(f"Event Detection F1: {event_type_f1*100:.1f}%")

        if "entities+non_entity_args" in overall:
            combined_f1 = overall["entities+non_entity_args"].get("f1", 0)
            summary_text.append(f"End-to-End F1: {combined_f1*100:.1f}%")

        # Count languages
        num_languages = len(
            [k for k in results.keys() if k not in ["all", "bucket_thresholds"]]
        )
        summary_text.append(f"Languages Evaluated: {num_languages}")

        summary_panel = Panel(
            "\n".join(summary_text),
            title="Summary",
            border_style="bright_blue",
        )

        console.print(summary_panel)


class MetricsFormatter:
    """Formats evaluation metrics in a human-readable way using rich tables."""

    def __init__(self) -> None:
        self.console = Console()

    def format_metrics(self, metrics: dict[str, dict[str, Any]]) -> None:
        """Format and display comprehensive metrics."""
        self.console.print("\n[bold blue]Evaluation Report[/bold blue]\n")

        # Display bucket thresholds first
        if "bucket_thresholds" in metrics:
            self._display_bucket_thresholds(metrics["bucket_thresholds"])

        # Separate languages from 'all' for better organization
        languages = {
            k: v for k, v in metrics.items() if k not in ["all", "bucket_thresholds"]
        }

        # Display per-language results
        if languages:
            self.console.print("\n[bold green]🌐 Per-Language Results[/bold green]")
            for lang, lang_metrics in languages.items():
                self._display_language_metrics(lang, lang_metrics)

        # Display overall results
        if "all" in metrics:
            self._display_overall_metrics(metrics["all"])

    def _display_bucket_thresholds(self, thresholds: dict[str, float]) -> None:
        """Display entity frequency bucket thresholds."""
        table = Table(title="Entity Frequency Bucket Thresholds", box=box.DOUBLE_EDGE)
        table.add_column("Bucket", style="cyan", no_wrap=True)
        table.add_column("Frequency Threshold", style="magenta", justify="right")

        for bucket, threshold in thresholds.items():
            table.add_row(bucket, f"{threshold:.1f}")

        self.console.print(table)

    def _display_language_metrics(self, language: str, metrics: dict[str, Any]) -> None:
        """Display metrics for a specific language."""

        # Main metrics table
        main_table = self._create_main_metrics_table(f"{language.upper()}", metrics)

        # Entity frequency bucket metrics - merged into one table
        if "buckets" in metrics:
            bucket_table = self._create_merged_bucket_table(
                f"Entity Frequency Effect on Entity Linking F1, for {language.upper()}",
                metrics["buckets"],
            )

            # Display main table
            self.console.print(main_table)

            # Display merged bucket table
            self.console.print(bucket_table)
        else:
            # Display main table only
            self.console.print(main_table)

        self.console.print()  # Add spacing

    def _display_overall_metrics(self, metrics: dict[str, Any]) -> None:
        """Display overall metrics with enhanced formatting."""
        self._display_language_metrics("All Languages", metrics)

    def _create_main_metrics_table(self, title: str, metrics: dict[str, Any]) -> Table:
        """Create the main metrics table for categories."""
        table = Table(title=title, box=box.HEAVY_EDGE)
        table.add_column("Category", style="cyan bold", no_wrap=True)
        table.add_column("F1", style="red", justify="right")
        table.add_column("Precision", style="green", justify="right")
        table.add_column("Recall", style="yellow", justify="right")
        table.add_column("TP", style="blue", justify="right")
        table.add_column("FP", style="orange3", justify="right")
        table.add_column("FN", style="purple", justify="right")

        # Define category display names and order
        category_names = {
            "event_type": "Event Detection",
            "entities": "All Entities",
            "seen_entities": "Seen Entities",
            "unseen_entities": "Unseen Entities",
            "generic_entities": "Generic Entities",
            "specific_entities": "Specific Entities",
            "non_entity_args": "Non-Entity Args",
            "entities+non_entity_args": "Entities + Non-Entity Args",
        }

        for category, display_name in category_names.items():
            if category in metrics:
                data = metrics[category]
                precision = self._format_percentage(data.get("precision", 0))
                recall = self._format_percentage(data.get("recall", 0))
                f1 = self._format_percentage(data.get("f1", 0))
                tp = str(int(data.get("tp", 0)))
                fp = str(int(data.get("fp", 0)))
                fn = str(int(data.get("fn", 0)))

                # Highlight high-performing metrics
                f1_style = "bold green" if data.get("f1", 0) > 0.8 else ""
                table.add_row(
                    display_name,
                    Text(f1, style=f1_style),
                    precision,
                    recall,
                    tp,
                    fp,
                    fn,
                )

        return table

    def _create_merged_bucket_table(
        self, title: str, buckets_data: dict[str, dict[str, Any]]
    ) -> Table:
        """Create a merged table showing all entity frequency buckets."""
        table = Table(title=title, box=box.ROUNDED)
        table.add_column("Entity Type", style="cyan bold", no_wrap=True)

        # Add columns for each bucket
        bucket_names = list(buckets_data.keys())
        for bucket_name in bucket_names:
            table.add_column(f"{bucket_name}\nF1", style="red", justify="right")

        entity_type_names = {
            "entities": "All Entities",
            "seen_entities": "Seen Entities",
            "unseen_entities": "Unseen Entities",
            "generic_entities": "Generic Entities",
            "specific_entities": "Specific Entities",
        }

        # Add rows for each entity type
        for entity_type, display_name in entity_type_names.items():
            row_data = [display_name]

            for bucket_name in bucket_names:
                bucket_data = buckets_data[bucket_name]
                if entity_type in bucket_data:
                    f1 = self._format_percentage(
                        bucket_data[entity_type].get("f1", 0), decimals=1
                    )
                    row_data.append(f1)
                else:
                    row_data.append("N/A")

            table.add_row(*row_data)

        return table

    def _create_bucket_table(
        self, bucket_name: str, bucket_data: dict[str, Any]
    ) -> Table:
        """Create a compact table for entity frequency bucket metrics."""
        table = Table(title=f"Entity Frequency Bucket: {bucket_name}", box=box.ROUNDED)
        table.add_column("Entity Type", style="cyan", no_wrap=True)
        table.add_column("F1", style="red", justify="right")
        table.add_column("P", style="green", justify="right")
        table.add_column("R", style="yellow", justify="right")

        entity_type_names = {
            "entities": "All",
            "seen_entities": "Seen",
            "unseen_entities": "Unseen",
            "generic_entities": "Generic",
            "specific_entities": "Specific",
        }

        for entity_type, display_name in entity_type_names.items():
            if entity_type in bucket_data:
                data = bucket_data[entity_type]
                f1 = self._format_percentage(data.get("f1", 0), decimals=1)
                precision = self._format_percentage(
                    data.get("precision", 0), decimals=1
                )
                recall = self._format_percentage(data.get("recall", 0), decimals=1)
                table.add_row(display_name, f1, precision, recall)

        return table

    def _format_percentage(self, value: float, decimals: int = 1) -> str:
        """Format a decimal as a percentage."""
        return f"{value * 100:.{decimals}f}%"
