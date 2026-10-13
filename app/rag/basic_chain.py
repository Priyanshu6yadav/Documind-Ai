from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.llm import get_llm

# step 1: Create the prompt templates
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "you are an AI teacher,"
        "Explain concept in simple English"
    ),
    (
        "human",
        "Explain {topic} with one simple example."
    )
])

# step 2: Initialize our local LLm
llm = get_llm()

# step 3: Create the output parser
parser = StrOutputParser()

# Step 4: Connect components using lcel
chain = prompt | llm | parser

# Step 5: Execute the chain
if __name__=="__main__":
    response = chain.invoke({
        "topic": "retrival augmented generation"
    })
    print(response)