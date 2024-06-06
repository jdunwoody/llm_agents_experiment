from transformers.agents import CodeAgent, PythonInterpreterTool


def _main():

    python_interpreter = PythonInterpreterTool()
    agent = CodeAgent(tools=[python_interpreter])
    agent.run("What is the result of 2 power 3.7384?")


if __name__ == "__main__":
    _main()
