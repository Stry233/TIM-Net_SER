from DeepSVDD.dataset.serDataset import SERDataset

def load_dataset(dataset_name, data_path, normal_class, need_tranform=True):
    """Loads the dataset."""

    implemented_datasets = ('IEMOCAP')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'IEMOCAP':
        dataset = SERDataset(root=data_path, normal_class=normal_class)

    return dataset
