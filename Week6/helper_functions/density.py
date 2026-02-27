from collections.abc import Callable, Iterable
from typing import Any, Literal, Union

import numpy as np

global available_kernels
available_kernels = Union[
    Literal[
        "gaussian", "exponential", "exponential_kernel", "multivariate_gaussian", "sinc"
    ],
    Callable,
]


def convolution_signed_measures(
    iterable_of_signed_measures,
    filtrations,
    bandwidth,
    flatten: bool = True,
    n_jobs: int = 1,
    backend="pykeops",
    kernel: available_kernels = "gaussian",
    **kwargs,
):
    """
    Evaluates the convolution of the signed measures Iterable(pts, weights) with a gaussian measure of bandwidth bandwidth, on a grid given by the filtrations

    Parameters
    ----------

     - iterable_of_signed_measures : (num_signed_measure) x [ (npts) x (num_parameters), (npts)]
     - filtrations : (num_parameter) x (filtration values)
     - flatten : bool
     - n_jobs : int

    Outputs
    -------

    The concatenated images, for each signed measure (num_signed_measures) x (len(f) for f in filtration_values)
    """
    from multipers.grids import todense

    grid_iterator = todense(filtrations, product_order=True)
    match backend:
        case "sklearn":

            def convolution_signed_measures_on_grid(
                signed_measures: Iterable[tuple[np.ndarray, np.ndarray]],
            ):
                return np.concatenate(
                    [
                        _pts_convolution_sparse_old(
                            pts=pts,
                            pts_weights=weights,
                            grid_iterator=grid_iterator,
                            bandwidth=bandwidth,
                            kernel=kernel,
                            **kwargs,
                        )
                        for pts, weights in signed_measures
                    ],
                    axis=0,
                )

        case "pykeops":

            def convolution_signed_measures_on_grid(
                signed_measures: Iterable[tuple[np.ndarray, np.ndarray]],
            ) -> np.ndarray:
                return np.concatenate(
                    [
                        _pts_convolution_pykeops(
                            pts=pts,
                            pts_weights=weights,
                            grid_iterator=grid_iterator,
                            bandwidth=bandwidth,
                            kernel=kernel,
                            **kwargs,
                        )
                        for pts, weights in signed_measures
                    ],
                    axis=0,
                )

            # compiles first once
            pts, weights = iterable_of_signed_measures[0][0]
            small_pts, small_weights = pts[:2], weights[:2]

            _pts_convolution_pykeops(
                small_pts,
                small_weights,
                grid_iterator=grid_iterator,
                bandwidth=bandwidth,
                kernel=kernel,
                **kwargs,
            )

    if n_jobs > 1 or n_jobs == -1:
        prefer = "processes" if backend == "sklearn" else "threads"
        from joblib import Parallel, delayed

        convolutions = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(convolution_signed_measures_on_grid)(sms)
            for sms in iterable_of_signed_measures
        )
    else:
        convolutions = [
            convolution_signed_measures_on_grid(sms)
            for sms in iterable_of_signed_measures
        ]
    if not flatten:
        out_shape = [-1] + [len(f) for f in filtrations]  # Degree
        convolutions = [x.reshape(out_shape) for x in convolutions]
    return np.asarray(convolutions)


# def _test(r=1000, b=0.5, plot=True, kernel=0):
# 	import matplotlib.pyplot  as plt
# 	pts, weigths = np.array([[1.,1.], [1.1,1.1]]), np.array([1,-1])
# 	pt_list = np.array(list(product(*[np.linspace(0,2,r)]*2)))
# 	img = _pts_convolution_sparse_pts(pts,weigths, pt_list,b,kernel=kernel)
# 	if plot:
# 		plt.imshow(img.reshape(r,-1).T, origin="lower")
# 		plt.show()


def _pts_convolution_sparse_old(
    pts: np.ndarray,
    pts_weights: np.ndarray,
    grid_iterator,
    kernel: available_kernels = "gaussian",
    bandwidth=0.1,
    **more_kde_args,
):
    """
    Old version of `convolution_signed_measures`. Scikitlearn's convolution is slower than the code above.
    """
    from sklearn.neighbors import KernelDensity

    if len(pts) == 0:
        # warn("Found a trivial signed measure !")
        return np.zeros(len(grid_iterator))
    kde = KernelDensity(
        kernel=kernel, bandwidth=bandwidth, rtol=1e-4, **more_kde_args
    )  # TODO : check rtol
    pos_indices = pts_weights > 0
    neg_indices = pts_weights < 0
    img_pos = (
        np.zeros(len(grid_iterator))
        if pos_indices.sum() == 0
        else kde.fit(
            pts[pos_indices], sample_weight=pts_weights[pos_indices]
        ).score_samples(grid_iterator)
    )
    img_neg = (
        np.zeros(len(grid_iterator))
        if neg_indices.sum() == 0
        else kde.fit(
            pts[neg_indices], sample_weight=-pts_weights[neg_indices]
        ).score_samples(grid_iterator)
    )
    return np.exp(img_pos) - np.exp(img_neg)


def _pts_convolution_pykeops(
    pts: np.ndarray,
    pts_weights: np.ndarray,
    grid_iterator,
    kernel: available_kernels = "gaussian",
    bandwidth=0.1,
    **more_kde_args,
):
    """
    Pykeops convolution
    """
    kde = KDE(kernel=kernel, bandwidth=bandwidth, **more_kde_args)
    return kde.fit(
        pts, sample_weights=np.asarray(pts_weights, dtype=pts.dtype)
    ).score_samples(np.asarray(grid_iterator, dtype=pts.dtype))


def gaussian_kernel(x_i, y_j, bandwidth):
    D = x_i.shape[-1]
    exponent = -(((x_i - y_j) / bandwidth) ** 2).sum(dim=-1) / 2
    # float is necessary for some reason (pykeops fails)
    kernel = (exponent).exp() / float((bandwidth*np.sqrt(2 * np.pi))**D) 
    return kernel


def multivariate_gaussian_kernel(x_i, y_j, covariance_matrix_inverse):
    # 1 / \sqrt(2 \pi^dim * \Sigma.det()) * exp( -(x-y).T @ \Sigma ^{-1} @ (x-y))
    # CF https://www.kernel-operations.io/keops/_auto_examples/pytorch/plot_anisotropic_kernels.html#sphx-glr-auto-examples-pytorch-plot-anisotropic-kernels-py
    #    and https://www.kernel-operations.io/keops/api/math-operations.html
    dim = x_i.shape[-1]
    z = x_i - y_j
    exponent = -(z.weightedsqnorm(covariance_matrix_inverse.flatten()) / 2)
    return (
        float((2 * np.pi) ** (-dim / 2))
        * (covariance_matrix_inverse.det().sqrt())
        * exponent.exp()
    )


def exponential_kernel(x_i, y_j, bandwidth):
    # 1 / \sigma * exp( norm(x-y, dim=-1))
    exponent = -(((((x_i - y_j) ** 2)).sum(dim=-1) ** 1 / 2) / bandwidth)
    kernel = exponent.exp() / bandwidth
    return kernel


def sinc_kernel(x_i, y_j, bandwidth):
    norm = ((((x_i - y_j) ** 2)).sum(dim=-1) ** 1 / 2) / bandwidth
    sinc = type(x_i).sinc
    kernel = 2 * sinc(2 * norm) - sinc(norm)
    return kernel


def _kernel(
    kernel: available_kernels = "gaussian",
):
    match kernel:
        case "gaussian":
            return gaussian_kernel
        case "exponential":
            return exponential_kernel
        case "multivariate_gaussian":
            return multivariate_gaussian_kernel
        case "sinc":
            return sinc_kernel
        case _:
            assert callable(
                kernel
            ), f"""
            --------------------------
            Unknown kernel {kernel}.
            -------------------------- 
            Custom kernel has to be callable, 
            (x:LazyTensor(n,1,D),y:LazyTensor(1,m,D),bandwidth:float) ---> kernel matrix

            Valid operations are given here:
            https://www.kernel-operations.io/keops/python/api/index.html
            """
            return kernel


# TODO : multiple bandwidths at once with lazy tensors
class KDE:
    """
    Simple kernel density estimation without pykeops, using numpy directly.
    """

    def __init__(
        self,
        bandwidth: Any = 1,
        kernel: available_kernels = "gaussian",
        return_log: bool = False,
    ):
        self.X = None
        self.bandwidth = bandwidth
        self.kernel = kernel
        self._sample_weights = None
        self.return_log = return_log

    def fit(self, X, sample_weights=None, y=None):
        self.X = X
        self._sample_weights = sample_weights
        return self

    def score_samples(self, Y, X=None, return_kernel=False):
        """Returns the kernel density estimates of each point in `Y`."""
        X = self.X if X is None else X
        if X.shape[0] == 0:
            return np.zeros((Y.shape[0]))
            
        # Compute pairwise distances
        if self.kernel == "gaussian":
            D = X.shape[1]
            diff = Y[:, np.newaxis, :] - X[np.newaxis, :, :]  # (Y_points, X_points, dims)
            sq_dist = (diff ** 2).sum(axis=2)  # (Y_points, X_points)
            kernel = np.exp(-sq_dist / (2 * self.bandwidth**2)) / ((self.bandwidth * np.sqrt(2 * np.pi))**D)
        elif self.kernel == "exponential":
            dist = np.sqrt(((Y[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2))
            kernel = np.exp(-dist / self.bandwidth) / self.bandwidth
        else:
            raise ValueError(f"Kernel {self.kernel} not implemented in simple version")

        if self._sample_weights is not None:
            kernel = kernel * self._sample_weights[np.newaxis, :]

        if return_kernel:
            return kernel
            
        density_estimation = kernel.mean(axis=1)  # average over X points
        
        return np.log(density_estimation) if self.return_log else density_estimation








# def _pts_convolution_sparse(pts:np.ndarray, pts_weights:np.ndarray, filtration_grid:Iterable[np.ndarray], kernel="gaussian", bandwidth=0.1, **more_kde_args):
# 	"""
# 	Old version of `convolution_signed_measures`. Scikitlearn's convolution is slower than the code above.
# 	"""
# 	from sklearn.neighbors import KernelDensity
# 	grid_iterator = np.asarray(list(product(*filtration_grid)))
# 	grid_shape = [len(f) for f in filtration_grid]
# 	if len(pts) == 0:
# 		# warn("Found a trivial signed measure !")
# 		return np.zeros(shape=grid_shape)
# 	kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, rtol = 1e-4, **more_kde_args) # TODO : check rtol

# 	pos_indices = pts_weights>0
# 	neg_indices = pts_weights<0
# 	img_pos = kde.fit(pts[pos_indices], sample_weight=pts_weights[pos_indices]).score_samples(grid_iterator).reshape(grid_shape)
# 	img_neg = kde.fit(pts[neg_indices], sample_weight=-pts_weights[neg_indices]).score_samples(grid_iterator).reshape(grid_shape)
# 	return np.exp(img_pos) - np.exp(img_neg)


# Precompiles the convolution
# _test(r=2,b=.5, plot=False)