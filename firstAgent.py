from dotenv import load_dotenv
load_dotenv()
import os

#--------------------------------------
# STEP 1: Initializing the Model
#--------------------------------------

from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="qwen/qwen-plus-2025-07-28:free",
    base_url="https://api.xkiro.com/v1",
    api_key=os.getenv("XKIRO_API_KEY")
)


#------------------------------------------
# STEP 2: Define your Tools for the Model
#------------------------------------------

# What LLM sees: The frameworks converts the function into a JSON Schema.
# {"name": "add", "description": "Add two numbers....", "parameters": {"a": {"type": "number"}, "b": {"type": "number"}}}

from langchain.tools import tool

@tool
def add(a:float, b:float) -> float:
    """Add two numbers together. Use this tool when you need to perform addition."""
    return a+b

@tool
def subtract(a:float, b:float) -> float:
    """Subtract two numbers. Use this tool when you need to perform subtraction."""
    return a - b

@tool
def multiply(a:float, b:float) -> float:
    """Multiply two numbers. Use this tool when you need to perform multiplication."""
    return a * b

@tool
def square(a: float) -> float:
    """Square a number. Use this tool when you need to perform squaring."""
    return a ** 2

# Put all tools into a list
tools = [add, subtract, multiply, square]


#--------------------------------------
# STEP 3: Create your Agent
#--------------------------------------

#create_agent builds afull ReAct loop:
# Reason -> Act (tool call) -> Observe -> Repeat ---Until the model is satisfied with the answer.

from langchain.agents import create_agent

agent = create_agent(
    model = model,
    tools = tools
)


#--------------------------------------
# STEP 4: Run your Agent
#--------------------------------------

# def run_agent(question : str):
#     """Run the agent and print the execution trace."""
#     print(f"User: {question}")
#     print("-" * 50)

#     result = agent.invoke({
#         "messages":[("user", question)]
#     })

#     print(f"Agent: {result}")


# def run_agent(question : str):
#     """Run the agent and print the execution trace."""
#     print(f"\nUser: {question}")
#     print("-" * 50)

#     result = agent.invoke({
#         "messages":[{"role": "user", "content": question}]
#     })
#     final_answer = result["messages"][-1].content

#     print(f"Agent: {final_answer}")


def run_agent(question: str):
    print("\n" + "=" * 60)
    print(f"USER: {question}")
    print("=" * 60)

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    messages = result["messages"]

    for message in messages:

        # LLM decision
        if hasattr(message, "tool_calls") and message.tool_calls:

            for call in message.tool_calls:
                tool_name = call["name"]
                args = call["args"]

                print("\n🤖 REASONING / DECISION:")
                print(f"   I need to use the '{tool_name}' tool.")
                print(f"   Arguments: {args}")

                print("\n🔧 ACTION:")
                print(f"   Calling {tool_name}({args})")

        # Tool result
        elif message.__class__.__name__ == "ToolMessage":

            print("\n📋 OBSERVATION:")
            print(f"   Tool returned: {message.content}")

        # Final answer
        elif message.__class__.__name__ == "AIMessage" and message.content:

            final_content = message.content

            # Gemini may return content as a list of blocks
            if isinstance(final_content, list):
                final_content = " ".join(
                    block["text"]
                    for block in final_content
                    if isinstance(block, dict) and block.get("type") == "text"
                )

            print("\n✅ FINAL ANSWER:")
            print(f"   {final_content}")

    print("\n" + "=" * 60)



#--------------------------------------
# STEP 5: Test your Agent
#--------------------------------------

# Test 1: Single tool call
run_agent("What is 42 + 58?")


# Test 2: Multiple tool calls in sequence
run_agent("What is 15 multiplied by 8, then subtracted by 10?")


# Test 3: Multi-step problem
run_agent(
    "I have a rectangle with width 12 and height 7. "
    "What is its area, and what is the square of that area?"
)