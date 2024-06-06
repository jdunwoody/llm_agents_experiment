from openai import OpenAI
from py_expression_eval import Parser
import re, time, os
from datetime import datetime


def stream_agent(system_prompt, tools, client, prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    def extract_action_and_input(text):
        action_pattern = r"Action: (.+?)\n"
        input_pattern = r"Action Input: \"(.+?)\""
        action = re.findall(action_pattern, text)
        action_input = re.findall(input_pattern, text)
        if action:
            action = action[0]

        return action, action_input

    while True:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            top_p=1,
        )
        response_text = response.choices[0].message.content
        print(response_text)

        action, action_input = extract_action_and_input(response_text)
        if action == "RESPONSE_TO_HUMAN":
            print(f"Response: {action_input[-1]}")
            break

        tool = tools.get(action, None)

        if not tool:
            raise Exception(f"Unknow tool for {action}")
        observation = tool(action_input[-1])
        print("Observation: ", observation)
        messages.extend(
            [
                {"role": "system", "content": response_text},
                {"role": "user", "content": f"Observation: {observation}"},
            ]
        )


def search(search_term):
    search_result = f"the search result: {search_term}"
    return search_result


def calculator(str):
    parser = Parser()
    return parser.parse(str).evaluate({})


def time(str):
    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")

    return formatted


def response_to_human(str):
    print(f"Response: {str}")


def _main():
    tools = {
        "SEARCH": search,
        "CALCULATOR": calculator,
        "TIME": time,
        "RESPONSE_TO_HUMAN": None,
    }
    system_prompt = f"""
        Answer the following questions and obey the following commands as best you can.

        You have access to the following tools:

            SEARCH: useful for when you need to answer questions about current events. You should ask targeted questions.
            CALCULATOR: Useful for when you need to answer questions about math. Use only a single python expression that can be evaluated by the python repl eg: 2 + 2
            TIME: Get the current time in a particular region
            RESPONSE_TO_HUMAN: When you need to respond to the human you are talking to.

        You will receive a message from the human, then you should start a loop and do one of two things

        Option 1: You use a tool to answer the question.
        For this, you should use the following format:
        Thought: you should always think about what to do
        Action: the action to take, should be one of {tools.keys()}
        Action Input: "the input to the action, to be sent to the tool"

        After this, the human will respond with an observation, and you will continue.

        Option 2: You respond to the human.
        For this, you should use the following format:
        Action: RESPONSE_TO_HUMAN
        Action Input: "your response to the human, summarizing what you did and what you learned"

        Begin!
    """

    client = OpenAI()

    result = stream_agent(
        system_prompt=system_prompt,
        tools=tools,
        client=client,
        prompt="What is the current time in Sydney, rounded to the nearest minute?",
    )

    print(result)


if __name__ == "__main__":
    _main()
