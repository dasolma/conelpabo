import importlib

from phm_framework.data import datasets as phmd

def get_task(meta, task, model_creator):
    task = phmd.get_task(meta, task)

    try:
        task_type = getattr(importlib.import_module(model_creator.__module__), 'TASK_TYPE')
        task['type'] = task_type
    except:
        pass


    return task
