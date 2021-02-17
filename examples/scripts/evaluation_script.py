import numpy as np

from datalib import Mnist
from datalib import Cifar10
from datalib import FashionMnist
from scriptify import scriptify
from time import time

from fgp import check
from fgp import CheckableModel
from fgp.certification import INCONCLUSIVE
from fgp.certification import NOT_ROBUST
from fgp.certification import ROBUST
from fgp.certification import TIMED_OUT


@scriptify
def script(
        dataset,
        architecture, 
        epsilon, 
        batch_size=32,
        keepgoing=False,
        timeout=120,
        madry=False,
        mmr=False,
        linf=False, 
        recap=False,
        seed=None, 
        samples=None):

    # Load the dataset.
    flat = not architecture.startswith('cnn')

    if dataset == 'mnist':
        data = Mnist(flat=flat)

    elif dataset == 'fmnist':
        data = FashionMnist(flat=flat)

    elif dataset == 'cifar':
        data = Cifar10()

    else:
        raise ValueError(f'invalid dataset: {dataset}')

    # Create the model.
    if flat:
        internal_neurons = [int(u) for u in architecture.split('.')]

        model = CheckableModel(
            data.input_shape, internal_neurons, data.num_classes)

        model.summary()

    else:
        channels = [int(u) for u in architecture.split('cnn')[1].split('.')]

        layer_details = []
        for nc in channels[:-1]:
            layer_details.append((nc, 4, 2, 'same'))

        layer_details.append(channels[-1])

        model = CheckableModel(
            data.input_shape, layer_details, data.num_classes)

        model.summary()

    # Load the model weights.
    if mmr:
        model_name = '{}-mmr-{}{}.h5'.format(
            dataset, 
            architecture, 
            '-l_inf' if linf else '')
      
    elif madry:
        model_name = '{}-{}{}.h5'.format(
            dataset, 
            architecture, 
            '-l_inf' if linf else '')

    else:
        raise ValueError(
            'specify training type by setting one of the following flags:\n'
            '  --madry\n'
            '  --mmr')

    model.load_weights(f'../models/{model_name}')

    print('\n>>>> successfully loaded model\n')

    # Compile for verification.
    model.compile_backprop('linf' if linf else 'l2')

    # Warm up.
    check(
        model, 
        data.x_tr[0],
        epsilon,
        timeout=10,
        batch_size=batch_size,
        keepgoing=keepgoing,
        recap=recap,
        cache_first_layer=True)

    # Get samples.
    if samples is None:
        with open('../indices/indices.txt', 'r') as indices_file:
            indices = [int(i) for i in indices_file.read().splitlines()]

    else:
        if seed is not None:
            np.random.seed(seed)

        indices = np.random.randint(0, len(data.x_te), samples)

    # Verify points.
    results = []
    result_times = []

    for i, j in enumerate(indices):
        print(f'==> point {i+1} / {len(indices)}')

        x = data.x_te[j]

        start_time = time()

        result = check(
            model, 
            x,
            epsilon,
            timeout=timeout,
            batch_size=batch_size,
            keepgoing=keepgoing,
            recap=recap,
            cache_first_layer=True)

        end_time = time()

        print(f'Checked in {end_time - start_time:.5f} seconds')

        result_times.append(end_time - start_time)
        results.append(result)

    result_times = np.array(result_times)
    results = np.array(results)

    # Gather results.
    correct_preds = (
        model.predict(data.x_te[indices]).argmax(axis=1) == data.y_te[indices])

    vra = (results[correct_preds] == ROBUST).sum() / float(len(indices))

    print(
        ('-'*79 + '\n') +
        'time\tdecided\tR\tNR\tUK\tTO\tVRA\n'
        f'{np.median(result_times):.3f}\t'
        f'{(results == ROBUST).mean() + (results == NOT_ROBUST).mean():.2f}\t'
        f'{(results == ROBUST).mean():.2f}\t'
        f'{(results == NOT_ROBUST).mean():.2f}\t'
        f'{(results == INCONCLUSIVE).mean():.2f}\t'
        f'{(results == TIMED_OUT).mean():.2f}\t'
        f'{vra:.2f}\n')
