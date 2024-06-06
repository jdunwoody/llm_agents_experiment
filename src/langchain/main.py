import os

import openai

# from langchain_aws import BedrockLLM
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import OpenAI

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.base import BaseCallbackHandler


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_end(self, event, context):
        print(f"Prompt: {event.prompt}")
        print(f"Response: {event.response}")


def _main():
    # langchain.debug = True
    # map_prompt = hub.pull("rlm/map-prompt")

    # message = map_prompt.messages[0]

    # print(message.input_variables)
    # print(message.template)

    # prompt_template = "What is the capital city of {country}?"
    # prompt = PromptTemplate(input_variables=["country"], template=prompt_template)
    # chain = prompt | llm
    # response = chain.invoke({"country": "Canada"})
    # print(response["text"])

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
    tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    print(tool.name)
    print(tool.description)
    print(tool.args)
    print(tool.return_direct)
    print(tool.run({"query": "Olivia Wilde"}))

    # prompt = hub.pull("hwchase17/react-chat-json")
    # prompt = hub.pull("hwchase17/react")
    prompt_str = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
    """
    prompt_template = ChatPromptTemplate.from_messages([("human", prompt_str)])

    llm = OpenAI(model_name="gpt-4-turbo", temperature=0)

    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = create_react_agent(prompt=prompt_template, tools=tools, llm=llm)

    agent_executor = AgentExecutor(agent=agent, tools=tools)
    query = "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?"
    result = agent_executor.invoke(input={"input": query})

    # result = agent.run(
    #     text="Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?"
    # )

    print(result)


if __name__ == "__main__":
    _main()
