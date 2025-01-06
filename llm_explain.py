import os
from dotenv import load_dotenv
load_dotenv()
from langchain import LLMChain, PromptTemplate
from langchain_groq import ChatGroq
import json

client = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),model="llama-3.3-70b-versatile"
)

def recommend(prompt):
    prompt_template = PromptTemplate(input_variables = ['prompt'],template = """You are an advanced language model who is the expert in analyzing human's character from a credit officer perspective. So understand the Human's intent and behaviour and only list recommendations to the credit officer backed by your evaluation. Format your response strictly as a json object with keys 'recommendations' and 'evaluation'
                                    User: {prompt}""" )
                                        
        
    chain = LLMChain(llm=client, prompt=prompt_template)
    response = chain.run(prompt)
    print(response)
    return json.loads(response[3:-3])