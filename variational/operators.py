import torch
import astra
import numpy as np

import math

class Operator(): 
    r"""
    The main class of the library. It defines the abstract Operator that will be subclassed for any specific case.
    """
    def __call__(self, x):
        return self._matvec(x)

    def __matmul__(self, x):
        return self._matvec(x)

    def T(self, x):
        return self._adjoint(x)

class CTProjector(Operator):    
    def __init__(self, img_shape, angles, det_size=None, geometry='parallel'):
        import astra
        
        super().__init__()
        # Input setup
        self.m, self.n = img_shape

        # Geometry
        self.geometry = geometry

        # Projector setup
        if det_size is None:
            self.det_size = int(max(self.n, self.m) * math.sqrt(2)) + 1
        else:
            self.det_size = det_size
        self.angles = angles
        self.n_angles = len(angles)

        # Define projector
        self.proj = self.get_astra_projection_operator()
        self.shape = self.proj.shape
        
    # ASTRA Projector
    def get_astra_projection_operator(self):
        # create geometries and projector
        if self.geometry == 'parallel':
            proj_geom = astra.create_proj_geom('parallel', 1.0, self.det_size, self.angles)
            vol_geom = astra.create_vol_geom(self.m, self.n)
            proj_id = astra.create_projector('linear', proj_geom, vol_geom)

        elif self.geometry == 'fanflat':
            proj_geom = astra.create_proj_geom('fanflat', 1.0, self.det_size, self.angles, 1800, 500)
            vol_geom = astra.create_vol_geom(self.m, self.n)
            proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
            
        else:
            print("Geometry (still) undefined.")
            return None

        return astra.OpTomo(proj_id)

    # On call, project
    def _matvec(self, x):
        y = self.proj @ x.flatten()
        return y
    
    def _adjoint(self, y):
        x = self.proj.T @ y.flatten()
        return x

    # FBP
    def FBP(self, y):
        x = self.proj.reconstruct('FBP', y.flatten())
        return x.reshape((self.m, self.n))