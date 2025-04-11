from sampling.sampling_fallacies_detection import run_fallacies

if __name__ == '__main__':
    run_fallacies(
        model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
        spl_name='spl2'
    )