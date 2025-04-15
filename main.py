from src.utils import get_config
from src.fallacies import run_fallacies
from src.aduc import run_aduc

def run(task: str=None):
    if task is not None:
        config = get_config(task)
        if task == 'fallacies':
            run_fallacies(**config)
        if task == 'aduc':
            run_aduc(**config)
    else:
        print(f'Error while getting config for task {task}')

if __name__ == '__main__':
    run('fallacies')