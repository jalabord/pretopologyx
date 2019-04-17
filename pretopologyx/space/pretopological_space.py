

# Here we define the class in charge of stocking the space and the method for the pseudoclure
# We need to define families of distance networks so the lsp can be more interesting
# We need to decide if we use prenetworks or neighborhoods
# dnf is a list of lists with integers between zero and len(neighborhoods)-1
class PretopologicalSpace:

    def __init__(self, prenetworks, dnf):
        self.prenetworks = prenetworks
        self.network_index = [i for i, prenetwork in enumerate(prenetworks) for item in prenetwork.thresholds]
        self.thresholds = [item for prenetwork in prenetworks for item in prenetwork.thresholds]
        self.dnf = dnf
        self.size = len(prenetworks[0].network) if prenetworks else 0

    def add_prenetworks(self, prenetworks):
        # we should check here the validity
        last_network_index = self.network_index[-1] if self.network_index else -1
        self.prenetworks += prenetworks
        self.network_index += [(i + last_network_index + 1) for i, prenetwork in enumerate(prenetworks)
                               for item in prenetwork.thresholds]
        self.thresholds += [item for prenetwork in prenetworks for item in prenetwork.thresholds]
        self.size = len(prenetworks[0].network) if not self.size else self.size

    def add_conjonction(self, conjonction):
        self.conjonctions.append(conjonction)


# Here we define the network_family class. This will allow us to calculate faster the pseudoclosure for a family
# of networks with the same weights but different thresholds, or with the same threshold and connections, but the
# weights are augmented or diminished by a constant factor.
# network is a list of numpy squared arrays, they all have to have the same size
# threshold is an array of floats, this way we can define many appartenence functions for a same network
class Prenetwork:
    def __init__(self, network, thresholds, weights=[1]):
        self.network = network
        self.thresholds = thresholds
        self.weights = weights
