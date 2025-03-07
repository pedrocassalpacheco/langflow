from langchain_core.vectorstores import VectorStore

from langflow.custom import Component
from langflow.io import Output


class GraphRAGSupportedMixin(Component):
    @classmethod
    def add_output(cls):
        if hasattr(cls, "outputs"):
            cls.outputs = cls.outputs.copy()  # Make a copy to avoid modifying the base class attribute
            output_names = [output.name for output in cls.outputs]
            if "vectorstoreconnection" not in output_names:
                cls.outputs.append(
                    Output(
                        display_name="Vector Store Connection", name="vectorstoreconnection", method="as_vector_store"
                    )
                )

    def as_vector_store(self) -> VectorStore:
        return self.build_vector_store()
