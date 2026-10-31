"""Step 0 on SageMaker: does the agent reach Claude on Bedrock and can it call a GoTrain tool?"""
from gotrain.agent import make_agent

agent = make_agent()
print(agent("Call gpu_status and list_tasks, then tell me in two lines what you can do."))
