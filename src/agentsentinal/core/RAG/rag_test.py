from . import rag_policy

rag = rag_policy.PolicyRAG()
rag.index(r"C:\Users\aweso\Desktop\Code\Hackathon\agentsentinal\src\agentsentinal\core\agents\inspector\SAMPLE-EMPLOYEE-POLICY-HANDBOOK.pdf")

question = "Is the agent allowed to share user data with third parties?"
context = rag.query_as_context(question)

print("\n--- Retrieved Policy Context ---")
print(context)

print("\n--- Example LLM Prompt ---")
print(f"""
You are a compliance overseer monitoring an AI agent.
The agent just performed an action. Check if it violates company policy.

Relevant policy sections:
{context}

Agent action: "Shared user email addresses with a marketing vendor."
Does this violate policy? Explain.
""")