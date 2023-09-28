import logging
import random
import time

import numpy as np
import torch

from DeepSVDD.DeepSvdd import DeepSVDD
from DeepSVDD.dataset import SERDataset

from DeepSVDD.dataset import load_dataset

def calculate_label_score(data, deepSVDD):
    """
    Calculate labels and scores for given data.

    Parameters:
    data (Tuple[torch.Tensor]): Tuple of inputs, labels and indices from the DataLoader.
        inputs (torch.Tensor): Input data.
        labels (torch.Tensor): Ground truth labels.
        idx (torch.Tensor): Indices of the data.
    deepSVDD (DeepSVDD): The neural network model to use for prediction.

    Returns:
    List[Tuple[int, int, float]]: List of tuples with indices, labels and calculated scores.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    R = torch.tensor(deepSVDD.R, device=device)  # radius R initialized with 0 by default.
    c = torch.tensor(deepSVDD.c, device=device) if deepSVDD.c is not None else None
    nu = deepSVDD.nu

    inputs, labels, idx = data
    inputs = inputs.to(device)
    net = deepSVDD.net.to(device)
    outputs = net(inputs)
    dist = torch.sum((outputs - c) ** 2, dim=1)

    if deepSVDD.objective == 'soft-boundary':
        scores = dist - R ** 2
    else:
        scores = dist

    return [idx.cpu().data.numpy().tolist()[0],
            labels.cpu().data.numpy().tolist()[0],
            scores.cpu().data.numpy().tolist()[0]]

def one_class_filter(input_shape, dataset, net_name, logger, xp_path="./DeepSVDD/models/",
                     objective='one-class', nu=0.1, device='cuda', seed=42,
                     optimizer_name='adam', lr=0.001, n_epochs=50, lr_milestone=None, batch_size=20,
                     weight_decay=1e-6, pretrain=True, ae_optimizer_name='adam', ae_lr=0.001,
                     ae_n_epochs=100, ae_lr_milestone=None, ae_batch_size=20, ae_weight_decay=1e-6,
                     n_jobs_dataloader=0):

    if lr_milestone is None:
        lr_milestone = [0]
    if ae_lr_milestone is None:
        ae_lr_milestone = [0]


    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info('Set seed to %d.' % seed)

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(objective, nu)
    deep_SVDD.set_network(net_name, input_shape)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('\n---Pretraining Start---')
        logger.info('Pretraining optimizer: %s' % ae_optimizer_name)
        logger.info('Pretraining learning rate: %g' % ae_lr)
        logger.info('Pretraining epochs: %d' % ae_n_epochs)
        logger.info('Pretraining learning rate scheduler milestones: %s' % (ae_lr_milestone,))
        logger.info('Pretraining batch size: %d' % ae_batch_size)
        logger.info('Pretraining weight decay: %g' % ae_weight_decay)

        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(input_shape, dataset,
                           optimizer_name=ae_optimizer_name,
                           lr=ae_lr,
                           n_epochs=ae_n_epochs,
                           lr_milestones=ae_lr_milestone,
                           batch_size=ae_batch_size,
                           weight_decay=ae_weight_decay,
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader)

    # Log training details
    logger.info('\n---Training Start---')
    logger.info('Training optimizer: %s' % optimizer_name)
    logger.info('Training learning rate: %g' % lr)
    logger.info('Training epochs: %d' % n_epochs)
    logger.info('Training learning rate scheduler milestones: %s' % (lr_milestone,))
    logger.info('Training batch size: %d' % batch_size)
    logger.info('Training weight decay: %g' % weight_decay)

    # Train model on dataset
    deep_SVDD.train(dataset,
                    optimizer_name=optimizer_name,
                    lr=lr,
                    n_epochs=n_epochs,
                    lr_milestones=lr_milestone,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    # deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # # Plot most anomalous and most normal (within-class) test samples
    # indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
    # indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)

    # Save results, model, and configuration
    # deep_SVDD.save_results(export_json=xp_path + '/results.json')
    # deep_SVDD.save_model(export_model=xp_path + '/model.tar')
    return deep_SVDD

def infer(dataset, deep_SVDD, threshold):

    # Get train data loader
    train_loader, test_loader = dataset.loaders(batch_size=1, shuffle_train=False, shuffle_test=False)

    eval_loader = train_loader

    print('---Start evaluation---')
    all_res = []
    with torch.no_grad():
        for data in eval_loader:
            content = calculate_label_score(data, deep_SVDD)
            all_res.append(content)  # idx, label, scores

    # Extract scores from all_res
    all_scores = [item[2] for item in all_res]

    # Normalize scores
    min_score = min(all_scores)
    max_score = max(all_scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in all_scores]

    return [idx for (inputs, labels, idx), score in zip(eval_loader, normalized_scores) if score <= threshold]


def filter_data(mask_index_train, mask_index_test, all_data, all_label, threshold=0.8):
    # Set up logging
    # print("train", mask_index_train)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = f'./DeepSVDD/log/log{time.time()}.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info('\n---Filtering Start---')
    logger.info('Log file is %s.' % log_file)
    logger.info("GPU is available." if torch.cuda.is_available() else "GPU is not available.")
    
    # generate the data cluster to be analyzed
    train_data = all_data[mask_index_train]
    train_label = all_label[mask_index_train]

    test_data = all_data[mask_index_test]
    test_label = all_label[mask_index_test]

    num_classes = len(set(tuple(row) for row in train_label))
    indices = []
    for cur_class in range(num_classes):
        logger.info(f'Start analyzing normal class: {cur_class} / {num_classes}')
        dataset = SERDataset(train_data, train_label, test_data, test_label, normal_class=cur_class)
        deepSvdd = one_class_filter((all_data.shape[1], all_data.shape[2]), dataset, 'general_cnn', logger)
        indices.append(infer(dataset, deepSvdd, threshold=threshold))
        # print(indices)

    filtered_x_index = sorted([int(t.item()) for sublist in indices for t in sublist])
    training_indices = [index for index in mask_index_train if index in filtered_x_index]

    # print(training_indices)

    return training_indices

