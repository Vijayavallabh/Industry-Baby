import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_groq import ChatGroq
import json

from typing import Optional

from typing_extensions import Annotated, TypedDict


class Output(TypedDict):
    """Recommendation output schema"""
    evaluation: Annotated[str, ..., "The evaluation of the recommendation"]
    recommendation: Annotated[str, ..., "The recommendation to the credit officer backed by the evaluation"]
    



client = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),model="llama-3.3-70b-versatile"
)

def recommend(prompt):
    messages = [
    SystemMessage( """You are an advanced language model who is the expert in analyzing human's character from a credit officer perspective. So understand the Human's intent and behaviour and only list recommendations to the credit officer backed by your evaluation."""),
    HumanMessage(prompt)]
    model = client.with_structured_output(Output)    
    response = model.invoke(messages)
    print(response)
    return response