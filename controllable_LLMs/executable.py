from .agents.example_agent import ExampleAgent

model = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q6_K_L"

example_agent = ExampleAgent(model)

output = example_agent.generate(model=model,system_prompt=example_agent.system(), prompt=example_agent.prompt(), schema=example_agent.schema())
print(output)