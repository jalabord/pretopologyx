import numpy as np
from pretopologyx.space.metrics import pseudoclosure

"""

For the context of pretopologies the algorithm has been presented under the exact same form, but instead of CC or SCC,
they used the pretopological equivalent of subspaces, the same happens for the articulation points, 1-AP and the weak
points.
Now, that presentation is not a problem in the context of graphs, where many great algorithms have been developed for
finding the biggest components, but to our knowlodge that hasen't been the case in petopology.
It may be iluminating to think about the incredible amount of resources that a naive implementation of the algorithm
would take.

La famille des composantes connexes (resp. fortement connexes) de (X,adh) est une partition de X (Belmandt)

We will take the subspaces step by step.

DEF: We say that a space (E, a(.)) of type V is strongly connected iff for all C subset ∈ P(E), C ≠ ø, F(C) = E.

DEF: We say that a space (E, a(.)) of type V is connected iff for all C subset ∈ P(E), C ≠ ø,
F(C) = E or F(X-F(C)) ∩ F(C) ≠ ø..

So if we have a set X that is strongly connected, and we add an element b, to show that X U {b} is still strongly
connected we only need to show (i) that adh(X) = X U {b} and (ii) that adh({b}) != {b}(We have shown that the conditions
are sufficient, not that they are necessary).
This shows that:
 - for all C subset of X, F(C) is now X U {b}. We knew the iterative application of adh on C and the respective images,
 eventually reached X, and we know now (because of (i) ) that adh(X)= X U {b}, so the adh will reach the whole space.
 - If adh({b}) != {b}, then there exist an element d in X, d != b, such that d belongs to adh({b}). Since we know that
 for every C of X, F(C) = X U {b}, then F({d}) = X U {b} subset of F({b, d}) subset of F(adh({b})) (because is a space
 of type V)
 - Finally we have that for all sets of the form A = C U {b}, where C subset of X, we have F(A) = X U {b},
 for the same argument used in the previous step.
So in order to find the connected components we just need to take any element, add another and test for (i) and (ii).
When we find an element that doesn't belong to the component, we put it in a new component, and every new element is
tested for each of those components, etc...


If we have a set X that is connected, and we add an element b, to show that X U {b} is still connected we only need
to show (i) that adh(X) = X U {b} or (ii) that adh({b}) != {b}
-If (i):
  - for all C subset of X,
        - if F(C) was X, now it's X U {b}.
        - if F(C) wasn't X, but F(X-F(C)) ∩ F(C) ≠ ø. then F((X U {b})-F(C)) ∩ F(C) ≠ ø, because (X U {b})-F(C) is
        bigger than F(X-F(C)) (space of type V)
  - for all sets of the form A = C U {b}, where C subset of X, either:
        - if F(C) was X, now it's X U {b}, so F(A) = X U {b}, because C is a subset of A and the space is of type V.
        - F(C) subset of F(A)
 eventually reached X, and we know now (because of (i) ) that adh(X)= X U {b}, so the adh will reach the whole space.
 - If adh({b}) != {b}, then there exist an element d in X, d != b, such that d belongs to adh({b}). Since we know that
 for every C of X, F(C) = X U {b}, then F({d}) = X U {b} subset of F({b, d}) subset of F(adh({b})) (because is a space
 of type V)
 - Finally we have that for all sets of the form A = C U {b}, where C subset of X, we have F(A) = X U {b},
 for the same argument used in the previous step.
So in order to find the connected components we just need to take any element, add another and test for (i) and (ii).
When we find an element that doesn't belong to the component, we put it in a new component, and every new element is
tested for each of those components, etc...


On montre (Z. Belmandt, Manuel de prétopologie et ses applications, op. cit.)
que la notion de composante est tout à fait compatible avec celle donnée en
théorie des graphes. Il suffit d’appliquer la prétopologie des descendants.
"""


def strongly_connected_subspace(pre_space):
    components = list()
    for i in np.arange(pre_space.size):
        new_element = np.zeros(pre_space.size)
        new_element[i] = 1
        new_component = True
        for compo in components:
            if not (pseudoclosure(new_element)*compo == np.zeros(pre_space.size)).all() and not \
                    (pseudoclosure(compo)*new_element == np.zeros(pre_space.size)).all():
                compo += new_element
                new_component = False
                break
        if new_component:
            components.append(new_element)


