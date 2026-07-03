from jackdaw.env import BalatroEnvironment, DirectAdapter
from jackdaw.env.agents import RandomAgent

# Initialize the 1:1 Python Balatro simulator
env = BalatroEnvironment(adapter_factory=DirectAdapter)
agent = RandomAgent()

obs, mask, info = env.reset()
done = False
steps = 0

print("Starting Balatro simulation...")

while not done and steps < 20:
    # Pass obs, mask, AND info to the agent
    action = agent.act(obs, mask, info)
    
    # Step the engine forward
    obs, terminated, truncated, mask, info = env.step(action)
    done = terminated or truncated
    steps += 1
    
    print(f"Step {steps} | Action Taken: {action} | Current Ante: {info.get('ante', 1)}")

print("Simulation finished successfully!")