"""Step 0 on SageMaker: does the agent reach Claude on Bedrock and can it call a GoTrain tool?"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # run from anywhere
from gotrain.agent import make_agent

agent = make_agent()
print(agent("Call gpu_status and list_tasks, then tell me in two lines what you can do."))
