from src.utils import get_config
from src.fallacies import run_fallacies
from src.aduc import run_aduc
from src.claim_detect import run_claim_detect
from src.evidence_detect import run_evidence_detect

def run(task: str=None):
    if task is not None:
        config = get_config(task)
        match task:
            case 'fallacies':
                run_fallacies(**config)
            case 'aduc':
                run_aduc(**config)
            case 'claim_detection':
                run_claim_detect(**config)
            case 'evidence_detection':
                run_evidence_detect(**config)
    else:
        print(f'Error while getting config for task {task}')

if __name__ == '__main__':
    run('fallacies')