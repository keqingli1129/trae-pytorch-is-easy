from transformers.pipelines import SUPPORTED_TASKS
from transformers import *
# for k, v in SUPPORTED_TASKS.items():
#     print(k, v) 

pipe = pipeline("text-classification")
print(pipe("I love you"))
