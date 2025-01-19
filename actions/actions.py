# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

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
from haystack.pipelines import Pipeline

# Load the Reader with the device set to "cpu"
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

# Add other components (e.g., Retriever)
pipeline = Pipeline()
pipeline.add_node(component=reader, name="Reader", inputs=["Query"])

# Run the pipeline
result = pipeline.run(query="Your query here", params={"Reader": {"top_k": 5}})
print(result)



from rasa_sdk import Action
from rasa_sdk.events import SlotSet
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader

class ActionQueryRAG(Action):
    def name(self):
        return "action_query_rag"

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get('text')
        pipeline = ExtractiveQAPipeline(reader, retriever)
        result = pipeline.run(query=query, params={"Retriever": {"top_k": 5}})
        answer = result["answers"][0].answer
        dispatcher.utter_message(text=answer)
        return []



