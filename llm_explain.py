import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

import json

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
def recommend(prompt):
    chat_completion = client.chat.completions.create(
        messages=[{'role':'system',
                'content':"""You are an advanced language model who is the expert in analyzing human's character from a credit officer perspective. So understand the Human's intent and behaviour and only list recommendations to the credit officer backed by your evaluation. Format your response strictly as a Dictionary object with keys 'recommendations' and 'evaluation'"""}
            ,{
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return json.loads(chat_completion.choices[0].message.content)