class Spheroid(FreeCellSimulation):
    """
    This uses free cells, i.e. cells that never share elements or nodes with other cells
    """

    def __init__(self, t0: float = 10, tg: float = 10, s: float = 10, sreg: float = 5,
                 seed: int = 49):
        """
        Object input parameters can be chosen as desired. These are the most useful ones for
        tuning behaviour and running tests
        
        :param t0: the pause phase duration
        :param tg: the growth phase duration
        :param s: the cell-cell interaction force law parameter used for both adhesion and repulsion
        :param sreg: the perimeter normalising force
        :param seed: seed for random number generator
        """
        pass
