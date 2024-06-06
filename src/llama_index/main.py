from pathlib import Path

import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import (
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.extractors.entity import EntityExtractor
from llama_index.llms.bedrock import Bedrock


def _main():
    nest_asyncio.apply()

    llm = Bedrock(model="amazon.titan-text-express-v1", profile_name="eliiza-sandbox")
    transformations = [
        SentenceSplitter(),
        TitleExtractor(nodes=5, llm=llm),
        QuestionsAnsweredExtractor(questions=3, llm=llm),
        SummaryExtractor(summaries=["prev", "self"], llm=llm),
        KeywordExtractor(keywords=10, llm=llm),
        EntityExtractor(prediction_threshold=0.5, llm=llm),
    ]
    pipeline = IngestionPipeline(transformations=transformations)

    pdf_path = Path(__file__).parents[1] / "data"
    uber_docs = SimpleDirectoryReader(
        input_files=[pdf_path / "uber_2021.pdf"]
    ).load_data()

    lyft_docs = SimpleDirectoryReader(
        input_files=[pdf_path / "lyft_2021.pdf"]
    ).load_data()

    uber_nodes = pipeline.run(documents=uber_docs)
    print(uber_nodes[1].metadata)

    lyft_nodes = pipeline.run(documents=lyft_docs)
    print(lyft_nodes[1].metadata)


if __name__ == "__main__":
    _main()
