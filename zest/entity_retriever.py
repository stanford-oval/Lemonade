from datetime import datetime
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tqdm import trange

from log_utils import logger


class SearchResultBlock(BaseModel):
    """
    A block of search result with additional similarity and probability scores.
    """

    document_title: str = Field(..., description="The title of the document")
    section_title: str = Field(
        ...,
        description="The hierarchical section title of the block excluding `document_title`, e.g. 'Land > Central Campus'. Section title can be empty, for instance the first section of Wikipedia articles.",
    )
    content: str = Field(
        ..., description="The content of the block, usually in Markdown format"
    )
    last_edit_date: Optional[datetime] = Field(
        None,
        description="The last edit date of the block in the format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS",
    )
    url: Optional[str] = Field(None, description="The URL of the block")

    num_tokens: int = Field(0, description="The number of tokens in the block content")
    block_metadata: Optional[dict] = Field(
        None, description="Additional metadata for the block"
    )

    similarity_score: float = Field(
        default=...,
        json_schema_extra={"example": 0.681},
        description="The similarity score between the search result and the query.",
    )
    probability_score: float = Field(
        default=...,
        json_schema_extra={"example": 0.331},
        description="Normalized similarity score. Can be viewed as the estimated probability that this SearchResultBlock is the answer to the query.",
    )
    summary: list[str] = Field(
        default_factory=list,
        json_schema_extra={"example": ["bullet point 1", "bullet point 2"]},
        description="Bullet points from the search result that are relevant to the query.",
    )

    model_config = ConfigDict(extra="forbid")  # Disallow extra fields


class QueryResult(BaseModel):
    results: list[SearchResultBlock] = Field(
        default_factory=list, description="The list of search results"
    )

    model_config = ConfigDict(extra="forbid")  # Disallow extra fields


class EntityRetriever:
    """Entity retrieval service with async HTTP client management."""

    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None

    async def setup(self) -> None:
        """Setup the entity retriever."""
        self.client = httpx.AsyncClient(timeout=120)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.client:
            await self.client.aclose()

    async def retrieve_entity_batch_via_search_api(
        self, batch_queries: list[str], num_results: int, retriever_endpoint: str
    ) -> list[dict]:
        """Retrieve entities in batch via search API."""
        results = [{"results": []}] * len(batch_queries)

        if not self.client:
            logger.error("HTTP client not initialized")
            return results

        try:
            response = await self.client.post(
                retriever_endpoint,
                json={
                    "query": batch_queries,
                    "rerank": False,
                    "num_blocks_to_rerank": num_results,
                    "num_blocks": num_results,
                },
            )

            if response.status_code == 429:
                logger.error(
                    "You have reached your rate limit for the retrieval server. Please wait and try later, or host your own retriever."
                )
                return results
            if response.status_code != 200:
                logger.error(
                    f"Error encountered when sending this request to retriever: {response.text}"
                )
                return results

            results = response.json()
            if len(results) != len(batch_queries):
                logger.warning(
                    f"Number of queries and results do not match. Length of queries: {len(batch_queries)}, Length of results: {len(results)}"
                )
                results = [{"results": []}] * len(batch_queries)

            return results
        except Exception as e:
            logger.error(f"Error retrieving entities: {e}")
            return results

    async def retrieve_entity_via_search_api(
        self,
        queries: list[str],
        num_results: int,
        batch_size: int,
        retriever_endpoint: str = "https://search.genie.stanford.edu/lemonade_entities",
    ) -> list[QueryResult]:
        """
        Retrieve search results from a retriever API.
        Args:
            queries: A list of queries to be sent to the retriever.
            retriever_endpoint: The endpoint URL of the retriever API.
            num_results: Number of blocks to return.
            batch_size: Number of queries to send in each batch.
        Returns:
            list[QueryResult]: A list of retrieved QueryResults.
        """
        ret = []
        for i in trange(
            0,
            len(queries),
            batch_size,
            desc="Retrieving entities via search API",
        ):
            batch_queries = queries[i : i + batch_size]
            results = await self.retrieve_entity_batch_via_search_api(
                batch_queries,
                num_results=num_results,
                retriever_endpoint=retriever_endpoint,
            )

            for j in range(len(batch_queries)):
                search_results = [
                    SearchResultBlock(**r) for r in results[j]["results"]
                ]  # convert to pydantic object
                ret.append(QueryResult(results=search_results))

        assert len(queries) == len(ret)
        return ret
