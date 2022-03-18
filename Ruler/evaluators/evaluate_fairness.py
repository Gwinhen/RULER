"""
This python file evaluates the discriminatory degree of models.
"""

import numpy as np
import generation_utilities


def ids_percentage(sample_round, num_gen, num_attribs, protected_attribs, constraint, model):
    # compute the percentage of individual discriminatory instances with 95% confidence
    statistics = np.empty(shape=(0,))
    for i in range(sample_round):
        gen_id = generation_utilities.purely_random(num_attribs, protected_attribs, constraint, model, num_gen)
        percentage = len(gen_id) / num_gen
        statistics = np.append(statistics, [percentage], axis=0)
    avg = np.average(statistics)
    std_dev = np.std(statistics)
    interval = 1.960 * std_dev / np.sqrt(sample_round)
    print('The percentage of individual discriminatory instances with .95 confidence:', avg, '±', interval)


def measure_discrimination(sample_round, num_gen, input_len, model, constraint, dataset):
    # measure the discrimination degree of models on each benchmark
    print(
        'Percentage of discriminatory instances for pgd-unfairness model:')
    if dataset == 'adult':
        for benchmark, protected_attribs in [('C-a', [0]), ('C-g', [7]), ('C-r', [6])]:
            print(benchmark, ':')
            ids_percentage(sample_round, num_gen, input_len, protected_attribs, constraint, model)
    elif dataset == 'bank':
        print(
            'Percentage of discriminatory instances for pgd-unfairness model:')
        print('B-a:')
        protected_attribs = [0]
        ids_percentage(sample_round, num_gen, input_len, protected_attribs, constraint, model)
    elif dataset == 'german':
        for benchmark, protected_attribs in [('G-g', [6]), ('G-a', [9])]:
            print(benchmark, ':')
            ids_percentage(sample_round, num_gen, input_len, protected_attribs, constraint, model)
    elif dataset == 'compas':
        print('Compas-r:')
        protected_attribs = [2]
        ids_percentage(sample_round, num_gen, input_len, protected_attribs, constraint, model)


# just for test
if __name__ == '__main__':
    pass
