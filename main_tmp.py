from typing import List, Union

import numpy as np

from tutorials.python_and_opengl.foo import vertices, edges, surfaces
from utils.tools import pyout

V: List[Union[List, str]] = [[]] * len(vertices)
for ii in range(len(vertices)):
    V[ii] = list(vertices[ii])

for ii, xyz in enumerate(V):
    for jj in range(3):
        v = str(xyz[jj])
        v = v.replace("1.7320508075688772", "sqrt(3)")
        v = v.replace("0.8660254037844386", "sqrt(3) / 2")

        V[ii][jj] = v
    V[ii] = ', '.join(V[ii])

E = [[]] * len(edges)
for ii in range(len(edges)):
    E[ii] = list(edges[ii])

S = [[]] * len(surfaces)
for ii in range(len(surfaces)):
    S[ii] = list(surfaces[ii])

with open("utils/tetrakaidecahedron.py", "w+") as f:
    f.write("import numpy as np\n")
    f.write("from numpy import sqrt\n\n")

    f.write(f"nodes = np.array([[{V[0]}],\n")
    for ii in range(1, len(V) - 1):
        f.write(f"                  [{V[ii]}],\n")
    f.write(f"                  [{V[-1]}]])\n\n")

    f.write(f"edges = np.array([{list(E[0])}, ")
    for ii in range(1, len(E) - 1):
        if ii % 8 == 0:
            f.write("\n                  ")
        f.write(f"{list(E[ii])}, ")
    f.write(f"{list(E[-1])}])\n\n")

    f.write(f"surfaces = np.array([{list(S[0])}, ")
    for ii in range(1, len(S) - 1):
        if ii % 5 == 0:
            f.write("\n                     ")
        f.write(f"{list(S[ii])}, ")
    f.write(f"{list(S[-1])}])\n\n")


def equals(tuple1, tuple2):
    # A = np.prod([v in tuple2 for v in tuple1])

    return np.prod(v in tuple2 for v in tuple1)

for s in S:
    dups= 0
    for s_ in S:
        if equals(s, s_) == 1:
            dups += 1
    if dups > 1:
        pyout(s)

# pyout()
