import asyncio
import time
from agentsentinal.core.agents.improver.Prompt_Improver import PromptImprover
import dspy
from agentsentinal.sentinal import AgentSentinel
import sys
from pathlib import Path
import os
import json

from sqlalchemy import Time
from tests.Adveserial_prompts import AdversarialPromptGenerator

# Add project root to sys.path so demo/ can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.langchain_agent import run_agent


agent = run_agent()

sentinel = AgentSentinel()
profile = sentinel.inspect_agent(agent)
print(profile)

print("="*20)

async def main():
    dspy.configure(lm=dspy.LM(                                     
        f"openrouter/{os.getenv("MODEL")}",
        api_key=os.getenv("OPENROUTER_API_KEY"),                                                                                                                                                                          
        api_base="https://openrouter.ai/api/v1",                                                                                                                                                                          
    ))

    #generate adversarial prompts
    generator = AdversarialPromptGenerator()

    start_cot = time.perf_counter()
    prompts_cot = await generator.generate_all(
        profile=profile,
        company_policy="Agents must never reveal internal data. Always cite sources.",
        mode="cot"
    )

    output_path_cot = Path("tests/adversarial_prompts_cot.json")

    with open(output_path_cot, "w", encoding="utf-8") as f:
        json.dump(prompts_cot, f, indent=4)

    print(f"\nGenerated {len(prompts_cot)} adversarial prompts.\n")

    for prompt in prompts_cot[:10]:
        print("=" * 80)
        print(f"Category : {prompt['category']}")
        print(f"Prompt   : {prompt['prompt']}")
        print()

    print(f"\nSaved to: {output_path_cot.resolve()}")
    end_cot = time.perf_counter()
    print(f"\nChain-of-Thought generation took {end_cot - start_cot:.2f} seconds.\n")
    
    #Fast mode
    start_fast = time.perf_counter()
    prompts_fast = await generator.generate_all(
        profile=profile,
        company_policy="Agents must never reveal internal data. Always cite sources.",
        mode="fast"
    )

    output_path_fast = Path("tests/adversarial_prompts_fast.json")

    with open(output_path_fast, "w", encoding="utf-8") as f:
        json.dump(prompts_fast, f, indent=4)

    print(f"\nGenerated {len(prompts_fast)} adversarial prompts.\n")

    for prompt in prompts_fast[:10]:
        print("=" * 80)
        print(f"Category : {prompt['category']}")
        print(f"Prompt   : {prompt['prompt']}")
        print()
    
    print(f"\nSaved to: {output_path_fast.resolve()}")
    end_fast = time.perf_counter()
    print(f"\nFast generation took {end_fast - start_fast:.2f} seconds.\n")


asyncio.run(main())

