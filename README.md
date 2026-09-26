# First LangChain AI Agent

This is my first AI Agent project, built while learning **LangChain**, **Google Gemini**, **tool calling**, and the **ReAct (Reason + Act) pattern**.

## What I Learned

- How to connect an LLM with LangChain
- How to create custom tools using Python functions
- How an LLM decides which tool to use
- How tool calls are executed by the agent
- How tool results are returned to the LLM
- How the ReAct flow works:
  
  `Reason → Action → Observation → Repeat → Final Answer`

## Tools Created

The agent currently has these tools:

- `add` – adds two numbers
- `subtract` – subtracts two numbers
- `multiply` – multiplies two numbers
- `square` – squares a number

## Tech Stack

- Python
- LangChain
- Google Gemini
- ReAct Agent Pattern

## Example

For the question:

> What is 15 multiplied by 8, then subtracted by 10?

The agent performs:

```text
Reason / Decision
      ↓
multiply(15, 8)
      ↓
Observation: 120
      ↓
Reason / Decision
      ↓
subtract(120, 10)
      ↓
Observation: 110
      ↓
Final Answer: 110