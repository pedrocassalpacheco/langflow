from typing import Any

from langchain_community.graph_vectorstores.extractors import HierarchyLinkExtractor, LinkExtractorTransformer
from langchain_core.documents import BaseDocumentTransformer

from langflow.base.document_transformers.model import LCDocumentTransformerComponent
from langflow.inputs import BoolInput, DataInput, StrInput


class PDFStructureExtractor(LCDocumentTransformerComponent):
    display_name = "PDF Structure Extractor"
    description = "Extract parent/child relationships from PDF documents parsed by Unstructured."
    documentation = "https://python.langchain.com/v0.2/api_reference/community/graph_vectorstores/langchain_community.graph_vectorstores.extractors.html_link_extractor.HtmlLinkExtractor.html"
    name = "PDFStructureExtractor"
    icon = "Unstructured"

    inputs = [
        DataInput(
            name="data_input",
            display_name="Input",
            info="The texts from which to extract links.",
            input_types=["Document", "Data"],
        ),
    ]

    def get_data_input(self) -> Any:
        return self.data_input

    def build_document_transformer(self) -> BaseDocumentTransformer:
        return LinkExtractorTransformer(
            [HierarchyLinkExtractor().as_document_extractor()]
        )
