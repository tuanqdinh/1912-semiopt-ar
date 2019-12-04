import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

class SFC:

    ################### Hilbert Codes ##################
    @staticmethod
    def decode_sfc(codes, M=3, p=10):
        """
            codes = [N, D]
        """
        hilbert_curve = HilbertCurve(p, M)
        num_points = 2 ** (M*p) - 1
        grid_size = 2**p - 1
        N = codes.shape[0]
        D = codes.shape[1]
        s = np.zeros((N, D, M))
        for i in range(N):
            for j in range(D):
                v = np.min((codes[i, j], 1))
                v = np.max((v, 0))
                v = int(v * num_points)
                coords = hilbert_curve.coordinates_from_distance(v)
                # p bins each coordinate
                s[i, j, :] = np.asarray(coords) / grid_size
        return s

    @staticmethod
    def encode_sfc(points, M=3, p=10):
        """
            points = [N, D, M]
        """
        hilbert_curve = HilbertCurve(p, M)
        num_points = 2 ** (M*p) - 1
        grid_size = 2**p - 1
        N = points.shape[0]
        D = points.shape[1]
        s = np.zeros((N, D))
        for i in range(N):
            for j in range(D):
                # change to integer
                coords = points[i, j, :] * grid_size + 0.5
                coords = coords.astype(int)
                dist = hilbert_curve.distance_from_coordinates(coords)
                s[i, j] = dist/num_points
        return s

    ################### Morton Codes ##################
    @staticmethod
    def get_grid_location(points, M=250):
        """
                        points = [N, D, 3]
                        p coordinates are in [0, 1]
                        Map to integral coordinate [250^3]
        """
        (N, D, _) = points.shape
        ipoints = np.zeros((N, D, 3))
        for i in range(N):
            for j in range(D):
                for k in range(3):
                    ipoints[i, j, k] = int(points[i, j, k] * M)
        return ipoints

    @staticmethod
    def get_morton_from_3d(ipoints):
        """
                        ipoints = [N, D, 3]
        """
        (N, D, _) = ipoints.shape
        codes = np.zeros((N, D))
        for i in range(N):
            for j in range(D):
                p = ipoints[i, j, :]
                codes[i, j] = pm.interleave3(int(p[0]), int(p[1]), int(p[2]))

        return codes

    @staticmethod
    def get_3d_from_morton(vs, M=250):
        codes = (vs * (M**3)).astype(int)
        # from IPython import embed; embed()
        (N, D) = codes.shape
        s = np.zeros((N, D, 3))
        for i in range(N):
            for j in range(D):
                mortoncode = int(codes[i, j])
                coord = pm.deinterleave3(mortoncode)
                for k in range(3):
                    s[i, j, k] = coord[k]/M
        return s

    @staticmethod
    def get_zcurve(points, M=250):
        """
                        points = [N, D, 3]
                        ipoints = [N, D, 3]
                        codes = [N, D]
                        seqs = [N, K^3]
        """
        (N, D, _) = points.shape
        ipoints = Helper.get_grid_location(points, M=M)
        codes = Helper.get_morton_from_3d(ipoints)
        return codes / (M**3)

    @staticmethod
    def get_binary_zcurve(points, M=250):
        """
                        points = [N, D, 3]
                        ipoints = [N, D, 3]
                        codes = [N, D]
                        seqs = [N, K^3]
        """
        (N, D, _) = points.shape
        ipoints = Helper.get_grid_location(points, M=M)
        codes = Helper.get_morton_from_3d(ipoints)
        # binary sequences
        seqs = np.zeros((N, M**3))
        for i in range(N):
            for j in range(D):
                p = codes[i, j]
                seqs[i, p] = 1

        return seqs

    ################# Hillbert ##########################
