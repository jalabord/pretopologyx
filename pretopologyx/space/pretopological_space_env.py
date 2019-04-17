# Here we define the class in charge of stocking the space and the method for the pseudoclosure
import numpy as np
from pretopologyx.space.pretopological_space import PretopologicalSpace


# It has an environment associated, i.e. a  matrix where the columns represents characteristics
# of the elements that are part of the environment. Useful for populations, for example.
# This allows to create networks that are a function of the characteristics of the elements.
class PretopologicalSpaceEnv(PretopologicalSpace):

    def __init__(self, prenetworks, dnf, environment, attribute_labels):
        super().__init__(prenetworks, dnf)
        self.env = environment
        self.attribute_labels = attribute_labels

    # We need to think better about this method for the general environment case
    # change in the neighborhoods product of the environment
    # the elements that have value -1 are not connected to anyone.
    def block_neighborhoods(self):
        pass


