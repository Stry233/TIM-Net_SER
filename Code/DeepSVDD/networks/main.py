from DeepSVDD.networks.generalCNN import GeneralCNN, GeneralCNN_Autoencoder
from DeepSVDD.networks.generalNet import GeneralNet, GeneralNet_Autoencoder


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('general_net', 'general_cnn')
    assert net_name in implemented_networks

    net = None

    if net_name == 'general_net':
        net = GeneralNet()

    if net_name == 'general_cnn':
        net = GeneralCNN()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('general_net', 'general_cnn')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'general_net':
        ae_net = GeneralNet_Autoencoder()

    if net_name == 'general_cnn':
        ae_net = GeneralCNN_Autoencoder()

    return ae_net
