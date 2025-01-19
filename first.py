from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.pipelines import ExtractiveQAPipeline

# Initialize document store
document_store = InMemoryDocumentStore()

# Load documents
docs = [
    {"content": "How to book an LPG cylinder?", "meta": {"language": "en"}},
    {"content": "एलपीजी सिलेंडर कैसे बुक करें?", "meta": {"language": "hi"}}
]
document_store.write_documents(docs)

# Initialize retriever
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-multiset-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-multiset-base"
)

# Update document embeddings
document_store.update_embeddings(retriever)










from haystack.nodes import FARMReader

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
pipeline = ExtractiveQAPipeline(reader, retriever)

query = "How can I book a gas cylinder?"
result = pipeline.run(query=query, params={"Retriever": {"top_k": 5}})
print(result)
