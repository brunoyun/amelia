from sampling.utils import get_config
from sampling.sampling_fallacies_detection import run_fallacies

if __name__ == '__main__':
    conf_fallacies = get_config('fallacies')
    run_fallacies(**conf_fallacies)