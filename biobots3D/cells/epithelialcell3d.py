from biobots3D.cells.abstractcell3d import AbstractCell3D


class EpithelialCell3D(AbstractCell3D):
    def __init__(self, pos=(0, 0, 0), face=(0, 1, 0)):
        super(EpithelialCell3D, self).__init__(pos, face)

