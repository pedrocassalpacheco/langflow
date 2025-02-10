import os

import orjson
from astrapy.admin import parse_api_endpoint

from langflow.helpers import docs_to_data
from langflow.inputs import (
    BoolInput,
    DictInput,
    DropdownInput,
    FloatInput,
    HandleInput,
    IntInput,
    SecretStrInput,
    StrInput,
)
from langflow.schema import Data
from langflow.custom import Component

class GraphRAGComponent(Component):
    display_name: str = "Graph RAG Retrieval"
    description: str = "Datastore Agnostic Graph RAG Retrieval"
    name = "graph_rag_retrieval"
    icon: str = "Globe"

    inputs = [
        HandleInput(
            name="embedding_model",
            display_name="Embedding Model",
            input_types=["Embeddings"],
            info="Specify the Embedding Model. Not required for Astra Vectorize collections.",
            required=False,
        ),
        HandleInput(
            name="vector_store",
            display_name="Vector Store Connection",
            input_types=["vector_store"],
            info="Connection to Vector Store.",
        ),
        MultilineInput(
            name="search_query",
            display_name="Search Query",
            tool_mode=True,
        ),
    ]
    
    def _build_search_args(self):
        args = {
            "k": self.number_of_results,
            "score_threshold": self.search_score_threshold,
        }

        if self.search_filter:
            clean_filter = {k: v for k, v in self.search_filter.items() if k and v}
            if len(clean_filter) > 0:
                args["filter"] = clean_filter
        return args

    def search_documents(self, vector_store=None) -> list[Data]:
        if not vector_store:
            vector_store = self.build_vector_store()

        self.log("Searching for documents in AstraDBGraphVectorStore.")
        self.log(f"Search query: {self.search_query}")
        self.log(f"Search type: {self.search_type}")
        self.log(f"Number of results: {self.number_of_results}")

        if self.search_query and isinstance(self.search_query, str) and self.search_query.strip():
            try:
                search_type = self._map_search_type()
                search_args = self._build_search_args()

                docs = vector_store.search(query=self.search_query, search_type=search_type, **search_args)

                # Drop links from the metadata. At this point the links don't add any value for building the
                # context and haven't been restored to json which causes the conversion to fail.
                self.log("Removing links from metadata.")
                for doc in docs:
                    if "links" in doc.metadata:
                        doc.metadata.pop("links")

            except Exception as e:
                msg = f"Error performing search in AstraDBGraphVectorStore: {e}"
                raise ValueError(msg) from e

            self.log(f"Retrieved documents: {len(docs)}")

            data = docs_to_data(docs)

            self.log(f"Converted documents to data: {len(data)}")

            self.status = data
            return data
        self.log("No search input provided. Skipping search.")
        return []

    def get_retriever_kwargs(self):
        search_args = self._build_search_args()
        return {
            "search_type": self._map_search_type(),
            "search_kwargs": search_args,
        }
