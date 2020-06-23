import numpy as np
from wickingpnm.model.network import PNM

class ArtPNM(PNM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(self, function, *args):
        graph = self.graph = function(*args)
        size = len(graph.nodes)
        print('Generated graph of size {}, filling with random data'.format(size))

        re = self.params['re'] = np.random.rand(size)
        h0e = self.params['h0e'] = np.random.rand(size)

        # re and h0e are on a scale of 1E-6 to 1E-3, waiting times from 1 to 1000
        # we will need a method to pass experimental information on the pore sizes in the future analogous to 'generate_waiting_times'
        for i in range(size):
            re[i] /= 10**np.random.randint(5, 6)
            h0e[i] /= 10**np.random.randint(4, 5)

        # define waiting times and inlets
        self.generate_waiting_times()
        self.build_inlets()
