# Here we define the class in charge of stocking the space and the method for the pseudoclosure
import numpy as np
import math
import svgwrite
from svgwrite.data.types import SVGAttribute
from svgwrite import cm, mm
from pretopologyx.space.pretopological_space_env import PretopologicalSpaceEnv
from pretopologyx.space.pretopological_space import Prenetwork


# It has a grid associated called env. blocked elements have value -1
# neighbors: list of bases, e.g. [[2,4],[1,3]]
class PretopologicalSpaceGrid(PretopologicalSpaceEnv):

    def __init__(self, environment, neighbors, dnf):
        super().__init__(list(), dnf, environment, "pixel_color")
        self.neighbors = neighbors
        self.size_r = environment.shape[0]
        self.size_c = environment.shape[1]
        self.create_prenetworks()
        self.block_neighborhoods()

    def create_prenetworks(self):
        for basis in self.neighbors:
            neigh = np.zeros((self.size_c*self.size_r, self.size_c*self.size_r))
            for i in basis:
                function_name = "create_n" + str(i)
                neigh += getattr(self, function_name)()
            self.add_prenetworks([Prenetwork(neigh, [1])])

    # change in the neighborhoods product of the environment
    # the elements that have value -1 are not connected to anyone.
    def block_neighborhoods(self):
        for prenetwork in self.prenetworks:
            for i, el in enumerate(self.env.flatten()):
                if el == -1:
                    prenetwork.network[i, :] = 0

    # blocked elements have value -1
    def block_environment(self, percent):
        already_blocked = len(np.argwhere(self.env == 1))
        needed = int(round((self.env.shape[0] ** 2) * (percent / 100))) - already_blocked
        options = np.argwhere(self.env == 0)
        idx = options[np.random.choice(len(options), needed, replace=False)]
        self.env[idx[:, 0], idx[:, 1]] = -1
        return self.env

    # modifies the environment to reflect the propagation
    def modify_environment(self, modification_set, new_value):
        idx = np.argwhere(modification_set == 1).flatten()
        self.env[np.unravel_index(idx, self.env.shape)] = new_value
        return self.env

    # neigh
    #    ^
    #     \
    #      el
    def create_n0(self):
        matrix_n0 = np.zeros((self.size_r*self.size_c, self.size_r*self.size_c))
        for i in range(1, self.size_r):
            for j in range(1, self.size_c):
                matrix_n0[i*self.size_c+j, i*self.size_c+j-self.size_c-1] = 1
        return matrix_n0

    # neigh <-- el
    def create_n1(self):
        matrix_n1 = np.zeros((self.size_r*self.size_c, self.size_r*self.size_c))
        for i in range(0, self.size_r):
            for j in range(1, self.size_c):
                matrix_n1[i*self.size_c+j,  i*self.size_c+j-1] = 1
        return matrix_n1

    #     el
    #    /
    #   v
    # neigh
    def create_n2(self):
        matrix_n2 = np.zeros((self.size_r*self.size_c, self.size_r*self.size_c))
        for i in range(0, self.size_r-1):
            for j in range(1, self.size_c):
                matrix_n2[i*self.size_c+j, i*self.size_c+j+self.size_c-1] = 1
        return matrix_n2

    # neigh
    #  ^
    #  |
    #  el
    def create_n3(self):
        matrix_n3 = np.zeros((self.size_r*self.size_c, self.size_r*self.size_c))
        for i in range(1, self.size_r):
            for j in range(0, self.size_c):
                matrix_n3[i*self.size_c+j, i*self.size_c+j-self.size_c] = 1
        return matrix_n3

    #  el
    #  |
    #  v
    # neigh
    def create_n4(self):
        matrix_n4 = np.zeros((self.size_r*self.size_c, self.size_r*self.size_c))
        for i in range(0, self.size_r-1):
            for j in range(0, self.size_c):
                matrix_n4[i*self.size_c+j, i*self.size_c+j+self.size_c] = 1
        return matrix_n4

    #      neigh
    #      ^
    #     /
    #  el
    def create_n5(self):
        matrix_n5 = np.zeros((self.size_r*self.size_c, self.size_r*self.size_c))
        for i in range(1, self.size_r):
            for j in range(0, self.size_c-1):
                matrix_n5[i*self.size_c+j, i*self.size_c+j-self.size_c+1] = 1
        return matrix_n5

    # el --> neigh
    def create_n6(self):
        matrix_n6 = np.zeros((self.size_r*self.size_c, self.size_r*self.size_c))
        for i in range(0, self.size_r):
            for j in range(0, self.size_c-1):
                matrix_n6[i*self.size_c+j, i*self.size_c+j+1] = 1
        return matrix_n6

    #  el
    #    \
    #     v
    #      neigh
    def create_n7(self):
        matrix_n7 = np.zeros((self.size_r*self.size_c, self.size_r*self.size_c))
        for i in range(0, self.size_r-1):
            for j in range(0, self.size_c-1):
                matrix_n7[i*self.size_c+j, i*self.size_c+j+self.size_c+1] = 1
        return matrix_n7


def draw_environment(env, dx=0, dy=0):

    svg = InkscapeDrawing('environment_lsp.svg', profile='full', size=(1600, 800))

    layer = svg.layer(label="Layer one")
    layer["sodipodi:insensitive"] = "true"
    svg.add(layer)

    for y, row in enumerate(env):
        for x, el in enumerate(row):
            color = 'green'
            if el == -1:
                color = 'gray'
            if el == 1:
                color = 'red'
            if el == 2:
                color = 'salmon'

            rect = svg.rect(insert=((x*11 + dx)*mm, (y*11 + dy)*mm), size=(10.4*mm, 10.4*mm), fill=color,
                            stroke=svgwrite.rgb(10, 10, 16, '%'), stroke_width='0')
            layer.add(rect)
    svg.save()


def draw_closures(number, list_closures, pre_space, initial_set, name, dy=0):
    print(number*math.sqrt(pre_space.size)*12)
    svg = InkscapeDrawing((name + '.svg'), profile='full', size=(number*math.sqrt(pre_space.size)*12*mm, math.sqrt(pre_space.size)*12*mm))

    layer = svg.layer(label="Layer one")
    layer["sodipodi:insensitive"] = "true"
    svg.add(layer)

    # We draw a closure
    list_indices = np.random.choice(len(list_closures), number, replace=False)

    for i, closure_index in enumerate(list_indices):
        # we reset the space
        i_set = np.zeros(pre_space.size)
        i_set[np.argwhere(pre_space.env.flatten() == 1).flatten()] = 1
        pre_space.modify_environment(i_set, 0)
        i_set = np.zeros(pre_space.size)
        i_set[np.argwhere(pre_space.env.flatten() == 2).flatten()] = 1
        pre_space.modify_environment(i_set, 0)


        # we change the colors of the seed and the closure
        pre_space.modify_environment(list_closures[closure_index], 1)
        i_set = np.zeros(pre_space.size)
        i_set[np.argwhere(initial_set == 1)[closure_index, 0]] = 1
        pre_space.modify_environment(i_set, 2)

        dx = i*math.sqrt(pre_space.size)*12
        for y, row in enumerate(pre_space.env):
            for x, el in enumerate(row):
                color = 'green'
                if el == -1:
                    color = 'gray'
                if el == 1:
                    color = 'red'
                if el == 2:
                    color = 'salmon'

                rect = svg.rect(insert=((x*11 + dx)*mm, (y*11 + dy)*mm), size=(10.4*mm, 10.4*mm), fill=color,
                                stroke=svgwrite.rgb(10, 10, 16, '%'), stroke_width='0')
                layer.add(rect)
        svg.save()


class InkscapeDrawing(svgwrite.Drawing):
    """An svgwrite.Drawing subclass which supports Inkscape layers"""
    INKSCAPE_NAMESPACE = 'http://www.inkscape.org/namespaces/inkscape'
    SODIPODI_NAMESPACE = 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'

    def __init__(self, *args, **kwargs):
        super(InkscapeDrawing, self).__init__(*args, **kwargs)

        inkscape_attributes = {
            'xmlns:inkscape': SVGAttribute('xmlns:inkscape',
                                           anim=False,
                                           types=[],
                                           const=frozenset([self.INKSCAPE_NAMESPACE])),
            'xmlns:sodipodi': SVGAttribute('xmlns:sodipodi',
                                           anim=False,
                                           types=[],
                                           const=frozenset([self.SODIPODI_NAMESPACE])),
            'inkscape:groupmode': SVGAttribute('inkscape:groupmode',
                                               anim=False,
                                               types=[],
                                               const=frozenset(['layer'])),
            'inkscape:label': SVGAttribute('inkscape:label',
                                           anim=False,
                                           types=frozenset(['string']),
                                           const=[]),
            'sodipodi:insensitive': SVGAttribute('sodipodi:insensitive',
                                                 anim=False,
                                                 types=frozenset(['string']),
                                                 const=[])
        }

        self.validator.attributes.update(inkscape_attributes)

        elements = self.validator.elements

        svg_attributes = set(elements['svg'].valid_attributes)
        svg_attributes.add('xmlns:inkscape')
        svg_attributes.add('xmlns:sodipodi')
        elements['svg'].valid_attributes = frozenset(svg_attributes)

        g_attributes = set(elements['g'].valid_attributes)
        g_attributes.add('inkscape:groupmode')
        g_attributes.add('inkscape:label')
        g_attributes.add('sodipodi:insensitive')
        elements['g'].valid_attributes = frozenset(g_attributes)

        self['xmlns:inkscape'] = self.INKSCAPE_NAMESPACE
        self['xmlns:sodipodi'] = self.SODIPODI_NAMESPACE

    def layer(self, **kwargs):
        """Create an inkscape layer.

        An optional 'label' keyword argument can be passed to set a user
        friendly name for the layer."""
        label = kwargs.pop('label', None)

        new_layer = self.g(**kwargs)
        new_layer['inkscape:groupmode'] = 'layer'

        if label:
            new_layer['inkscape:label'] = label

        return new_layer
