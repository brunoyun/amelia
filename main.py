from src.utils import get_config
from src.fallacies import run_fallacies
from src.aduc import run_aduc

def run(task: str=None):
    config = get_config(task)
    all_task = {
        'fallacies': run_fallacies(**config),
        'aduc': run_aduc(**config)
    }
    all_task.get(task)

if __name__ == '__main__':
    # conf_fallacies = get_config('fallacies')
    # run_fallacies(**conf_fallacies)
    run('aduc')