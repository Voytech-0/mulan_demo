# Author: Leland McInnes <leland.mcinnes@gmail.com>
# File taken from UMAP but converted to jax
# License: BSD 3 clause
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from sklearn.metrics import pairwise_distances

_mock_identity = jnp.eye(2, dtype=jnp.float32)
_mock_cost = 1.0 - _mock_identity
_mock_ones = jnp.ones(2, dtype=jnp.float32)


@jax.jit
def sign(a):
    return jnp.where(a < 0, -1, 1)


@jax.jit
def euclidean(x, y):
    r"""Standard euclidean distance.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    return jnp.sqrt(jnp.sum((x - y) ** 2))


@jax.jit
def euclidean_grad(x, y):
    r"""Standard euclidean distance and its gradient.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
        \frac{dD(x, y)}{dx} = (x_i - y_i)/D(x,y)
    """
    d = jnp.sqrt(jnp.sum((x - y) ** 2))
    grad = (x - y) / (1e-6 + d)
    return d, grad


@jax.jit
def standardised_euclidean(x, y, sigma=None):
    r"""Euclidean distance standardised against a vector of standard
    deviations per coordinate.

    ..math::
        D(x, y) = \sqrt{\sum_i \frac{(x_i - y_i)**2}{v_i}}
    """
    if sigma is None:
        sigma = jnp.ones_like(x)
    return jnp.sqrt(jnp.sum(((x - y) ** 2) / sigma))


@jax.jit
def standardised_euclidean_grad(x, y, sigma=None):
    r"""Euclidean distance standardised against a vector of standard
    deviations per coordinate with gradient.

    ..math::
        D(x, y) = \sqrt{\sum_i \frac{(x_i - y_i)**2}{v_i}}
    """
    if sigma is None:
        sigma = jnp.ones_like(x)
    d = jnp.sqrt(jnp.sum((x - y) ** 2 / sigma))
    grad = (x - y) / (1e-6 + d)
    return d, grad


@jax.jit
def manhattan(x, y):
    r"""Manhattan, taxicab, or l1 distance.

    ..math::
        D(x, y) = \sum_i |x_i - y_i|
    """
    return jnp.sum(jnp.abs(x - y))


@jax.jit
def manhattan_grad(x, y):
    r"""Manhattan, taxicab, or l1 distance with gradient.

    ..math::
        D(x, y) = \sum_i |x_i - y_i|
    """
    result = jnp.sum(jnp.abs(x - y))
    grad = jnp.sign(x - y)
    return result, grad


@jax.jit
def chebyshev(x, y):
    r"""Chebyshev or l-infinity distance.

    ..math::
        D(x, y) = \max_i |x_i - y_i|
    """
    return jnp.max(jnp.abs(x - y))


@jax.jit
def chebyshev_grad(x, y):
    r"""Chebyshev or l-infinity distance with gradient.

    ..math::
        D(x, y) = \max_i |x_i - y_i|
    """
    abs_diff = jnp.abs(x - y)
    result = jnp.max(abs_diff)
    max_i = jnp.argmax(abs_diff)
    grad = jnp.zeros_like(x)
    grad = grad.at[max_i].set(jnp.sign(x[max_i] - y[max_i]))
    return result, grad


@jax.jit
def minkowski(x, y, p=2):
    r"""Minkowski distance.

    ..math::
        D(x, y) = \left(\sum_i |x_i - y_i|^p\right)^{\frac{1}{p}}

    This is a general distance. For p=1 it is equivalent to
    manhattan distance, for p=2 it is Euclidean distance, and
    for p=infinity it is Chebyshev distance. In general it is better
    to use the more specialised functions for those distances.
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += jnp.abs(x[i] - y[i]) ** p
    return result ** (1.0 / p)


@jax.jit
def minkowski_grad(x, y, p=2):
    r"""Minkowski distance with gradient.

    ..math::
        D(x, y) = \left(\sum_i |x_i - y_i|^p\right)^{\frac{1}{p}}

    This is a general distance. For p=1 it is equivalent to
    manhattan distance, for p=2 it is Euclidean distance, and
    for p=infinity it is Chebyshev distance. In general it is better
    to use the more specialised functions for those distances.
    """
    # Compute Minkowski distance
    diff = jnp.abs(x - y)
    result = jnp.sum(diff ** p)
    # Compute gradient
    # grad[i] = (|x[i] - y[i]|**(p-1)) * sign(x[i] - y[i]) * result**(1/(p-1))
    # Note: result**(1/(p-1)) is only valid for p != 1
    grad = (diff ** (p - 1.0)) * jnp.sign(x - y) * (result ** (1.0 / (p - 1.0)))
    return result ** (1.0 / p), grad.astype(jnp.float32)


@jax.jit
def poincare(u, v):
    r"""Poincare distance.

    ..math::
        \delta (u, v) = 2 \frac{ \lVert  u - v \rVert ^2 }{ ( 1 - \lVert  u \rVert ^2 ) ( 1 - \lVert  v \rVert ^2 ) }
        D(x, y) = \operatorname{arcosh} (1+\delta (u,v))
    """
    sq_u_norm = jnp.sum(u * u)
    sq_v_norm = jnp.sum(v * v)
    sq_dist = jnp.sum(jnp.power(u - v, 2))
    return jnp.arccosh(1 + 2 * (sq_dist / ((1 - sq_u_norm) * (1 - sq_v_norm))))

def hyperboloid(x, y):
    return hyperboloid_grad(x, y)[0]

@jax.jit
def hyperboloid_grad(x, y):
    s = jnp.sqrt(1 + jnp.sum(x ** 2))
    t = jnp.sqrt(1 + jnp.sum(y ** 2))

    B = s * t
    for i in range(x.shape[0]):
        B -= x[i] * y[i]

    B = jax.lax.cond(B <= 1, lambda: 1.0 + 1e-8, lambda: B)

    grad_coeff = 1.0 / (jnp.sqrt(B - 1) * jnp.sqrt(B + 1))

    # return jnp.arccosh(B), jnp.zeros(x.shape[0])

    grad = jnp.zeros(x.shape[0])
    for i in range(x.shape[0]):
        grad = grad.at[i].set(grad_coeff * (((x[i] * t) / s) - y[i]))

    return jnp.arccosh(B), grad


@jax.jit
def weighted_minkowski(x, y, w=_mock_ones, p=2):
    r"""A weighted version of Minkowski distance.

    ..math::
        D(x, y) = \left(\sum_i w_i |x_i - y_i|^p\right)^{\frac{1}{p}}

    If weights w_i are inverse standard deviations of data in each dimension
    then this represented a standardised Minkowski distance (and is
    equivalent to standardised Euclidean distance for p=1).
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += w[i] * jnp.abs(x[i] - y[i]) ** p
    return result ** (1.0 / p)


@jax.jit
def weighted_minkowski_grad(x, y, w=_mock_ones, p=2):
    r"""A weighted version of Minkowski distance with gradient.

    ..math::
        D(x, y) = \left(\sum_i w_i |x_i - y_i|^p\right)^{\frac{1}{p}}

    If weights w_i are inverse standard deviations of data in each dimension
    then this represented a standardised Minkowski distance (and is
    equivalent to standardised Euclidean distance for p=1).
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += w[i] * (jnp.abs(x[i] - y[i])) ** p

    grad = jnp.zeros(x.shape[0], dtype=jnp.float32)
    for i in range(x.shape[0]):
        grad = grad.at[i].set(
            w[i]
            * jnp.pow(jnp.abs(x[i] - y[i]), (p - 1.0))
            * jnp.sign(x[i] - y[i])
            * jnp.pow(result, (1.0 / (p - 1)))
        )
    return result ** (1.0 / p), grad


@jax.jit
def mahalanobis(x, y, vinv=_mock_identity):
    result = 0.0

    diff = jnp.empty(x.shape[0], dtype=jnp.float32)

    for i in range(x.shape[0]):
        diff = diff.at[i].set(x[i] - y[i])

    for i in range(x.shape[0]):
        tmp = 0.0
        for j in range(x.shape[0]):
            tmp += vinv[i, j] * diff[j]
        result += diff[i] * tmp

    return jnp.sqrt(result)


@jax.jit
def mahalanobis_grad(x, y, vinv=_mock_identity):
    result = 0.0

    diff = jnp.empty(x.shape[0], dtype=jnp.float32)

    for i in range(x.shape[0]):
        diff = diff.at[i].set(x[i] - y[i])

    grad_tmp = jnp.zeros(x.shape)
    for i in range(x.shape[0]):
        tmp = 0.0
        for j in range(x.shape[0]):
            tmp += vinv[i, j] * diff[j]
        grad_tmp = grad_tmp.at[i].set(tmp)
        result += tmp * diff[i]
    dist = jnp.sqrt(result)
    grad = grad_tmp / (1e-6 + dist)
    return dist, grad


@jax.jit
def hamming(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        if x[i] != y[i]:
            result += 1.0

    return float(result) / x.shape[0]


@jax.jit
def canberra(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        denominator = jnp.abs(x[i]) + jnp.abs(y[i])
        if denominator > 0:
            result += jnp.abs(x[i] - y[i]) / denominator

    return result


@jax.jit
def canberra_grad(x, y):
    result = 0.0
    grad = jnp.zeros(x.shape)
    for i in range(x.shape[0]):
        denominator = jnp.abs(x[i]) + jnp.abs(y[i])
        if denominator > 0:
            result += jnp.abs(x[i] - y[i]) / denominator
            grad = grad.at[i].set(
                jnp.sign(x[i] - y[i]) / denominator
                - jnp.abs(x[i] - y[i]) * jnp.sign(x[i]) / denominator ** 2
                + jnp.sign(x[i] - y[i]) / denominator
                - jnp.abs(x[i] - y[i]) * jnp.sign(y[i]) / denominator ** 2
            )
    return result, grad


@jax.jit
def bray_curtis(x, y):
    numerator = 0.0
    denominator = 0.0
    for i in range(x.shape[0]):
        numerator += jnp.abs(x[i] - y[i])
        denominator += jnp.abs(x[i] + y[i])

    if denominator > 0.0:
        return float(numerator) / denominator
    else:
        return 0.0


@jax.jit
def bray_curtis_grad(x, y):
    numerator = 0.0
    denominator = 0.0
    for i in range(x.shape[0]):
        numerator += jnp.abs(x[i] - y[i])
        denominator += jnp.abs(x[i] + y[i])

    if denominator > 0.0:
        dist = float(numerator) / denominator
        grad = (jnp.sign(x - y) - dist) / denominator
        return dist, grad
    else:
        return 0.0, jnp.zeros(x.shape)
        dist = 0.0
        grad = jnp.zeros(x.shape)

    return dist, grad


@jax.jit
def jaccard(x, y):
    num_non_zero = 0.0
    num_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_non_zero += x_true or y_true
        num_equal += x_true and y_true

    if num_non_zero == 0.0:
        return 0.0
    else:
        return 1.0 - float(num_equal) / num_non_zero
        return float(num_non_zero - num_equal) / num_non_zero


@jax.jit
def matching(x, y):
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return float(num_not_equal) / x.shape[0]


@jax.jit
def dice(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return 1.0 - 2.0 * float(num_true_true) / (2.0 * num_true_true + num_not_equal)
        return num_not_equal / (2.0 * num_true_true + num_not_equal)


@jax.jit
def kulsinski(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0:
        return 0.0
    else:
        return (num_not_equal - num_true_true + x.shape[0]) / (
                num_not_equal + x.shape[0]
        )


@jax.jit
def rogers_tanimoto(x, y):
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return (2.0 * num_not_equal) / (x.shape[0] + num_not_equal)


@jax.jit
def russellrao(x, y):
    num_true_true = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true

    if num_true_true == jnp.sum(x != 0) and num_true_true == jnp.sum(y != 0):
        return 0.0
    else:
        return 1.0 - float(num_true_true) / x.shape[0]
        return float(x.shape[0] - num_true_true) / (x.shape[0])


@jax.jit
def sokal_michener(x, y):
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return (2.0 * num_not_equal) / (x.shape[0] + num_not_equal)


@jax.jit
def sokal_sneath(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return 1.0 - float(num_true_true) / (num_true_true + 0.5 * num_not_equal)
        return num_not_equal / (0.5 * num_true_true + num_not_equal)


@jax.jit
def haversine(x, y):
    if x.shape[0] != 2:
        raise ValueError("haversine is only defined for 2 dimensional data")
    sin_lat = jnp.sin(0.5 * (x[0] - y[0]))
    sin_long = jnp.sin(0.5 * (x[1] - y[1]))
    result = jnp.sqrt(sin_lat ** 2 + jnp.cos(x[0]) * jnp.cos(y[0]) * sin_long ** 2)
    return 2.0 * jnp.arcsin(result)


@jax.jit
def haversine_grad(x, y):
    # spectral initialization puts many points near the poles
    # currently, adding pi/2 to the latitude avoids problems
    # TODO: reimplement with quaternions to avoid singularity

    if x.shape[0] != 2:
        raise ValueError("haversine is only defined for 2 dimensional data")
    sin_lat = jnp.sin(0.5 * (x[0] - y[0]))
    cos_lat = jnp.cos(0.5 * (x[0] - y[0]))
    sin_long = jnp.sin(0.5 * (x[1] - y[1]))
    cos_long = jnp.cos(0.5 * (x[1] - y[1]))

    a_0 = jnp.cos(x[0] + jnp.pi / 2) * jnp.cos(y[0] + jnp.pi / 2) * sin_long ** 2
    a_1 = a_0 + sin_lat ** 2

    # Clamp a_1 to [0, 1] for numerical stability
    a_1_clamped = jnp.clip(jnp.abs(a_1), 0.0, 1.0)
    d = 2.0 * jnp.arcsin(jnp.sqrt(a_1_clamped))
    denom = jnp.sqrt(jnp.abs(a_1 - 1)) * jnp.sqrt(jnp.abs(a_1))
    grad = jnp.array(
        [
            (
                    sin_lat * cos_lat
                    - jnp.sin(x[0] + jnp.pi / 2) * jnp.cos(y[0] + jnp.pi / 2) * sin_long ** 2
            ),
            (jnp.cos(x[0] + jnp.pi / 2) * jnp.cos(y[0] + jnp.pi / 2) * sin_long * cos_long),
        ]
    ) / (denom + 1e-6)
    return d, grad


@jax.jit
def yule(x, y):
    num_true_true = 0.0
    num_true_false = 0.0
    num_false_true = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_true_false += x_true and (not y_true)
        num_false_true += (not x_true) and y_true

    num_false_false = x.shape[0] - num_true_true - num_true_false - num_false_true

    if num_true_false == 0.0 or num_false_true == 0.0:
        return 0.0
    else:
        return (2.0 * num_true_false * num_false_true) / (
                num_true_true * num_false_false + num_true_false * num_false_true
        )


@jax.jit
def cosine(x, y):
    x1_norm = jnp.maximum(jnp.linalg.norm(x, axis=-1), 1e-20)
    y_norm = jnp.maximum(jnp.linalg.norm(y, axis=-1), 1e-20)
    return 1. - jnp.sum(x * y, -1) / x1_norm / y_norm


@jax.jit
def cosine_grad(x, y):
    result = jnp.sum(x * y)
    norm_x = jnp.sum(x ** 2)
    norm_y = jnp.sum(y ** 2)

    def both_zero_case():
        dist = 0.0
        grad = jnp.zeros_like(x)
        return dist, grad

    def one_zero_case():
        dist = 1.0
        grad = jnp.zeros_like(x)
        return dist, grad

    def normal_case():
        dist = 1.0 - (result / jnp.sqrt(norm_x * norm_y))
        grad = -(x * result - y * norm_x) / (jnp.sqrt(norm_x ** 3 * norm_y) + 1e-8)
        return dist, grad

    return jax.lax.cond(
        jnp.logical_and(norm_x == 0.0, norm_y == 0.0),
        both_zero_case,
        lambda: jax.lax.cond(
            jnp.logical_or(norm_x == 0.0, norm_y == 0.0),
            one_zero_case,
            normal_case
        )
    )


@jax.jit
def correlation(x, y):
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif dot_product == 0.0:
        return 1.0
    else:
        return 1.0 - (dot_product / jnp.sqrt(norm_x * norm_y))


@jax.jit
def hellinger(x, y):
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0

    for i in range(x.shape[0]):
        result += jnp.sqrt(x[i] * y[i])
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0 and l1_norm_y == 0:
        return 0.0
    elif l1_norm_x == 0 or l1_norm_y == 0:
        return 1.0
    else:
        return jnp.sqrt(1 - result / jnp.sqrt(l1_norm_x * l1_norm_y))


@jax.jit
def hellinger_grad(x, y):
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0

    grad_term = jnp.empty(x.shape[0])

    for i in range(x.shape[0]):
        grad_term = grad_term.at[i].set(jnp.sqrt(x[i] * y[i]))
        result += grad_term[i]
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0 and l1_norm_y == 0:
        dist = 0.0
        grad = jnp.zeros(x.shape)
    elif l1_norm_x == 0 or l1_norm_y == 0:
        dist = 1.0
        grad = jnp.zeros(x.shape)
    else:
        dist_denom = jnp.sqrt(l1_norm_x * l1_norm_y)
        dist = jnp.sqrt(1 - result / dist_denom)
        grad_denom = 2 * dist
        grad_numer_const = (l1_norm_y * result) / (2 * dist_denom ** 3)

        grad = (grad_numer_const - (y / grad_term * dist_denom)) / grad_denom

    return dist, grad


@jax.jit
def approx_log_Gamma(x):
    if x == 1:
        return 0
    # x2= 1/(x*x);
    return x * jnp.log(x) - x + 0.5 * jnp.log(2.0 * jnp.pi / x) + 1.0 / (x * 12.0)
    # + x2*(-1.0/360.0) + x2* (1.0/1260.0 + x2*(-1.0/(1680.0)  +\
    #  x2*(1.0/1188.0 + x2*(-691.0/360360.0 + x2*(1.0/156.0 +\
    #  x2*(-3617.0/122400.0 + x2*(43687.0/244188.0 + x2*(-174611.0/125400.0) +\
    #  x2*(77683.0/5796.0 + x2*(-236364091.0/1506960.0 + x2*(657931.0/300.0))))))))))))


@jax.jit
def log_beta(x, y):
    a = min(x, y)
    b = max(x, y)
    if b < 5:
        value = -jnp.log(b)
        for i in range(1, int(a)):
            value += jnp.log(i) - jnp.log(b + i)
        return value
    else:
        return approx_log_Gamma(x) + approx_log_Gamma(y) - approx_log_Gamma(x + y)


@jax.jit
def log_single_beta(x):
    return jnp.log(2.0) * (-2.0 * x + 0.5) + 0.5 * jnp.log(2.0 * jnp.pi / x) + 0.125 / x


@jax.jit
def ll_dirichlet(data1, data2):
    """The symmetric relative log likelihood of rolling data2 vs data1
    in n trials on a die that rolled data1 in sum(data1) trials.

    ..math::
        D(data1, data2) = DirichletMultinomail(data2 | data1)
    """

    n1 = jnp.sum(data1)
    n2 = jnp.sum(data2)

    def body_fun(i, vals):
        log_b, self_denom1, self_denom2 = vals
        cond = data1[i] * data2[i] > 0.9
        log_b = jax.lax.select(
            cond,
            log_b + log_beta(data1[i], data2[i]),
            log_b
        )
        self_denom1 = jax.lax.select(
            cond | (data1[i] > 0.9),
            self_denom1 + log_single_beta(data1[i]),
            self_denom1
        )
        self_denom2 = jax.lax.select(
            cond | (data2[i] > 0.9),
            self_denom2 + log_single_beta(data2[i]),
            self_denom2
        )
        return (log_b, self_denom1, self_denom2)

    log_b, self_denom1, self_denom2 = jax.lax.fori_loop(
        0, data1.shape[0], body_fun, (0.0, 0.0, 0.0)
    )

    return jnp.sqrt(
        1.0 / n2 * (log_b - log_beta(n1, n2) - (self_denom2 - log_single_beta(n2)))
        + 1.0 / n1 * (log_b - log_beta(n2, n1) - (self_denom1 - log_single_beta(n1)))
    )


@jax.jit
def symmetric_kl(x, y, z=1e-11):  # pragma: no cover
    r"""
    symmetrized KL divergence between two probability distributions

    ..math::
        D(x, y) = \frac{D_{KL}\left(x \Vert y\right) + D_{KL}\left(y \Vert x\right)}{2}
    """
    n = x.shape[0]
    x_sum = 0.0
    y_sum = 0.0
    kl1 = 0.0
    kl2 = 0.0

    for i in range(n):
        x = x.at[i].set(x[i] + z)
        x_sum += x[i]
        y = y.at[i].set(y[i] + z)
        y_sum += y[i]

    for i in range(n):
        x = x.at[i].set(x[i] / x_sum)
        y = y.at[i].set(y[i] / y_sum)

    for i in range(n):
        kl1 += x[i] * jnp.log(x[i] / y[i])
        kl2 += y[i] * jnp.log(y[i] / x[i])

    return (kl1 + kl2) / 2


@jax.jit
def symmetric_kl_grad(x, y, z=1e-11):  # pragma: no cover
    """
    symmetrized KL divergence and its gradient

    """
    n = x.shape[0]
    x_sum = 0.0
    y_sum = 0.0
    kl1 = 0.0
    kl2 = 0.0

    for i in range(n):
        x = x.at[i].set(x[i] + z)
        x_sum += x[i]
        y = y.at[i].set(y[i] + z)
        y_sum += y[i]

    for i in range(n):
        x = x.at[i].set(x[i] / x_sum)
        y = y.at[i].set(y[i] / y_sum)

    for i in range(n):
        kl1 += x[i] * jnp.log(x[i] / y[i])
        kl2 += y[i] * jnp.log(y[i] / x[i])

    dist = (kl1 + kl2) / 2
    grad = (jnp.log(y / x) - (x / y) + 1) / 2

    return dist, grad


@jax.jit
def correlation_grad(x, y):
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y

    if norm_x == 0.0 and norm_y == 0.0:
        dist = 0.0
        grad = jnp.zeros(x.shape)
    elif dot_product == 0.0:
        dist = 1.0
        grad = jnp.zeros(x.shape)
    else:
        dist = 1.0 - (dot_product / jnp.sqrt(norm_x * norm_y))
        grad = ((x - mu_x) / norm_x - (y - mu_y) / dot_product) * dist

    return dist, grad


@jax.jit
def sinkhorn_distance(
        x, y, M=_mock_identity, cost=_mock_cost, maxiter=64
):  # pragma: no cover
    p = (x / x.sum()).astype(jnp.float32)
    q = (y / y.sum()).astype(jnp.float32)

    u = jnp.ones(p.shape, dtype=jnp.float32)
    v = jnp.ones(q.shape, dtype=jnp.float32)

    def body_fun(_, vals):
        u, v = vals
        t = M @ v
        u = jnp.where(t > 0, p / t, u)
        t = M.T @ u
        v = jnp.where(t > 0, q / t, v)
        return (u, v)

    u, v = jax.lax.fori_loop(0, maxiter, body_fun, (u, v))

    pi = jnp.diag(v) @ M @ jnp.diag(u)
    result = jnp.sum(jnp.where(pi > 0, pi * cost, 0.0))
    return result


@jax.jit
def spherical_gaussian_energy_grad(x, y):  # pragma: no cover
    mu_1 = x[0] - y[0]
    mu_2 = x[1] - y[1]

    sigma = jnp.abs(x[2]) + jnp.abs(y[2])
    sign_sigma = jnp.sign(x[2])

    dist = (mu_1 ** 2 + mu_2 ** 2) / (2 * sigma) + jnp.log(sigma) + jnp.log(2 * jnp.pi)
    grad = jnp.empty(3, jnp.float32)

    grad = grad.at[0].set(mu_1 / sigma)
    grad = grad.at[1].set(mu_2 / sigma)
    grad = grad.at[2].set(sign_sigma * (1.0 / sigma - (mu_1 ** 2 + mu_2 ** 2) / (2 * sigma ** 2)))

    return dist, grad


@jax.jit
def diagonal_gaussian_energy_grad(x, y):  # pragma: no cover
    mu_1 = x[0] - y[0]
    mu_2 = x[1] - y[1]

    sigma_11 = jnp.abs(x[2]) + jnp.abs(y[2])
    sigma_12 = 0.0
    sigma_22 = jnp.abs(x[3]) + jnp.abs(y[3])

    det = sigma_11 * sigma_22
    sign_s1 = jnp.sign(x[2])
    sign_s2 = jnp.sign(x[3])

    def grad_fallback():
        return mu_1 ** 2 + mu_2 ** 2, jnp.array([0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)

    def grad_main():
        cross_term = 2 * sigma_12
        m_dist = (
                jnp.abs(sigma_22) * (mu_1 ** 2)
                - cross_term * mu_1 * mu_2
                + jnp.abs(sigma_11) * (mu_2 ** 2)
        )
        dist = (m_dist / det + jnp.log(jnp.abs(det))) / 2.0 + jnp.log(2 * jnp.pi)
        grad = jnp.empty(6, dtype=jnp.float32)
        grad = grad.at[0].set((2 * sigma_22 * mu_1 - cross_term * mu_2) / (2 * det))
        grad = grad.at[1].set((2 * sigma_11 * mu_2 - cross_term * mu_1) / (2 * det))
        grad = grad.at[2].set(sign_s1 * (sigma_22 * (det - m_dist) + det * mu_2 ** 2) / (2 * det ** 2))
        grad = grad.at[3].set(sign_s2 * (sigma_11 * (det - m_dist) + det * mu_1 ** 2) / (2 * det ** 2))
        return dist, grad

    return jax.lax.cond(det == 0.0, grad_fallback, grad_main)

@jax.jit
def gaussian_energy(x, y):
    return gaussian_energy_grad(x, y)[0]

@jax.jit
def gaussian_energy_grad(x, y):  # pragma: no cover
    mu_1 = x[0] - y[0]
    mu_2 = x[1] - y[1]

    # Ensure width are positive
    x2 = jnp.abs(x[2])
    y2 = jnp.abs(y[2])

    # Ensure heights are positive
    x3 = jnp.abs(x[3])
    y3 = jnp.abs(y[3])

    # Ensure angle is in range -pi,pi
    x4 = jnp.arcsin(jnp.sin(x[4]))
    y4 = jnp.arcsin(jnp.sin(y[4]))

    # Covariance entries for y
    a = y2 * jnp.cos(y4) ** 2 + y3 * jnp.sin(y4) ** 2
    b = (y2 - y3) * jnp.sin(y4) * jnp.cos(y4)
    c = y3 * jnp.cos(y4) ** 2 + y2 * jnp.sin(y4) ** 2

    # Sum of covariance matrices
    sigma_11 = x2 * jnp.cos(x4) ** 2 + x3 * jnp.sin(x4) ** 2 + a
    sigma_12 = (x2 - x3) * jnp.sin(x4) * jnp.cos(x4) + b
    sigma_22 = x2 * jnp.sin(x4) ** 2 + x3 * jnp.cos(x4) ** 2 + c

    # Determinant of the sum of covariances
    det_sigma = jnp.abs(sigma_11 * sigma_22 - sigma_12 ** 2)
    x_inv_sigma_y_numerator = (
            sigma_22 * mu_1 ** 2 - 2 * sigma_12 * mu_1 * mu_2 + sigma_11 * mu_2 ** 2
    )

    def grad_fallback():
        return (
            mu_1 ** 2 + mu_2 ** 2,
            jnp.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=jnp.float32),
        )

    def grad_main():
        dist = x_inv_sigma_y_numerator / det_sigma + jnp.log(det_sigma) + jnp.log(2 * jnp.pi)
        grad = jnp.zeros(5, jnp.float32)
        grad = grad.at[0].set((2 * sigma_22 * mu_1 - 2 * sigma_12 * mu_2) / det_sigma)
        grad = grad.at[1].set((2 * sigma_11 * mu_2 - 2 * sigma_12 * mu_1) / det_sigma)

        grad2 = (
                mu_2 * (mu_2 * jnp.cos(x4) ** 2 - mu_1 * jnp.cos(x4) * jnp.sin(x4))
                + mu_1 * (mu_1 * jnp.sin(x4) ** 2 - mu_2 * jnp.cos(x4) * jnp.sin(x4))
        )
        grad2 *= det_sigma
        grad2 -= x_inv_sigma_y_numerator * jnp.cos(x4) ** 2 * sigma_22
        grad2 -= x_inv_sigma_y_numerator * jnp.sin(x4) ** 2 * sigma_11
        grad2 += x_inv_sigma_y_numerator * 2 * sigma_12 * jnp.sin(x4) * jnp.cos(x4)
        grad2 /= det_sigma ** 2 + 1e-8
        grad = grad.at[2].set(grad2)

        grad3 = (
                mu_1 * (mu_1 * jnp.cos(x4) ** 2 - mu_2 * jnp.cos(x4) * jnp.sin(x4))
                + mu_2 * (mu_2 * jnp.sin(x4) ** 2 - mu_1 * jnp.cos(x4) * jnp.sin(x4))
        )
        grad3 *= det_sigma
        grad3 -= x_inv_sigma_y_numerator * jnp.sin(x4) ** 2 * sigma_22
        grad3 -= x_inv_sigma_y_numerator * jnp.cos(x4) ** 2 * sigma_11
        grad3 -= x_inv_sigma_y_numerator * 2 * sigma_12 * jnp.sin(x4) * jnp.cos(x4)
        grad3 /= det_sigma ** 2 + 1e-8
        grad = grad.at[3].set(grad3)

        grad4 = (x3 - x2) * (
                2 * mu_1 * mu_2 * jnp.cos(2 * x4) - (mu_1 ** 2 - mu_2 ** 2) * jnp.sin(2 * x4)
        )
        grad4 *= det_sigma
        grad4 -= x_inv_sigma_y_numerator * (x3 - x2) * jnp.sin(2 * x4) * sigma_22
        grad4 -= x_inv_sigma_y_numerator * (x2 - x3) * jnp.sin(2 * x4) * sigma_11
        grad4 -= x_inv_sigma_y_numerator * 2 * sigma_12 * (x2 - x3) * jnp.cos(2 * x4)
        grad4 /= det_sigma ** 2 + 1e-8
        grad = grad.at[4].set(grad4)

        return dist, grad

    return jax.lax.cond(det_sigma < 1e-32, grad_fallback, grad_main)


@jax.jit
def spherical_gaussian_grad(x, y):  # pragma: no cover
    mu_1 = x[0] - y[0]
    mu_2 = x[1] - y[1]

    sigma = x[2] + y[2]
    sigma_sign = jnp.sign(sigma)

    def grad_fallback():
        return 10.0, jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)

    def grad_main():
        dist = (
                (mu_1 ** 2 + mu_2 ** 2) / jnp.abs(sigma)
                + 2 * jnp.log(jnp.abs(sigma))
                + jnp.log(2 * jnp.pi)
        )
        grad = jnp.empty(3, dtype=jnp.float32)
        grad = grad.at[0].set((2 * mu_1) / jnp.abs(sigma))
        grad = grad.at[1].set((2 * mu_2) / jnp.abs(sigma))
        grad = grad.at[2].set(sigma_sign * (-(mu_1 ** 2 + mu_2 ** 2) / (sigma ** 2) + (2 / jnp.abs(sigma))))
        return dist, grad

    return jax.lax.cond(sigma == 0, grad_fallback, grad_main)


# Special discrete distances -- where x and y are objects, not vectors


def get_discrete_params(data, metric):
    if metric == "ordinal":
        return {"support_size": float(data.max() - data.min()) / 2.0}
    elif metric == "count":
        min_count = scipy.stats.tmin(data)
        max_count = scipy.stats.tmax(data)
        lambda_ = scipy.stats.tmean(data)
        normalisation = count_distance(min_count, max_count, poisson_lambda=lambda_)
        return {
            "poisson_lambda": lambda_,
            "normalisation": normalisation / 2.0,  # heuristic
        }
    elif metric == "string":
        lengths = np.array([len(x) for x in data])
        max_length = scipy.stats.tmax(lengths)
        max_dist = max_length / 1.5  # heuristic
        normalisation = max_dist / 2.0  # heuristic
        return {"normalisation": normalisation, "max_dist": max_dist / 2.0}  # heuristic

    else:
        return {}


@jax.jit
def categorical_distance(x, y):
    return jnp.where(x == y, 0.0, 1.0)


@jax.jit
def hierarchical_categorical_distance(x, y, cat_hierarchy=[{}]):
    n_levels = float(len(cat_hierarchy))
    for level, cats in enumerate(cat_hierarchy):
        if cats[x] == cats[y]:
            return float(level) / n_levels
    else:
        return 1.0


@jax.jit
def ordinal_distance(x, y, support_size=1.0):
    return jnp.abs(x - y) / support_size


@jax.jit
def count_distance(x, y, poisson_lambda=1.0, normalisation=1.0):
    lo = jnp.minimum(x, y).astype(int)
    hi = jnp.maximum(x, y).astype(int)

    log_lambda = jnp.log(poisson_lambda)

    def log_k_factorial_fn(lo):
        def body_fun(k, val):
            return val + jnp.log(k)

        return jax.lax.fori_loop(2, lo, body_fun, 0.0)

    log_k_factorial = jax.lax.cond(
        lo < 2,
        lambda: 0.0,
        lambda: jax.lax.cond(
            lo < 10,
            lambda: log_k_factorial_fn(lo),
            lambda: approx_log_Gamma(lo + 1),
        ),
    )

    def result_body(k, val):
        return val + k * log_lambda - poisson_lambda - log_k_factorial + jnp.log(k)

    result = jax.lax.fori_loop(lo, hi, result_body, 0.0)

    return result / normalisation


@jax.jit
def levenshtein(x, y, normalisation=1.0, max_distance=20):
    x_len, y_len = len(x), len(y)

    # Opt out of some comparisons
    if jnp.abs(x_len - y_len) > max_distance:
        return jnp.abs(x_len - y_len) / normalisation

    v0 = jnp.arange(y_len + 1).astype(jnp.float32)
    v1 = jnp.zeros(y_len + 1)

    def outer_body(i, vals):
        v0, v1 = vals
        v1 = v1.at[i].set(i + 1)

        def inner_body(j, v1_):
            deletion_cost = v0[j + 1] + 1
            insertion_cost = v1_[j] + 1
            substitution_cost = jnp.where(x[i] == y[j], 0, 1)
            v1_ = v1_.at[j + 1].set(jnp.minimum(jnp.minimum(deletion_cost, insertion_cost), substitution_cost))
            return v1_

        v1 = jax.lax.fori_loop(0, y_len, inner_body, v1)
        v0 = v1
        return v0, v1

    v0, v1 = jax.lax.fori_loop(0, x_len, outer_body, (v0, v1))

    # Abort early if we've already exceeded max_dist
    if jnp.min(v0) > max_distance:
        return max_distance / normalisation

    return v0[y_len] / normalisation


named_distances = {
    # general minkowski distances
    "euclidean": euclidean,
    "l2": euclidean,
    "manhattan": manhattan,
    "taxicab": manhattan,
    "l1": manhattan,
    "chebyshev": chebyshev,
    "linfinity": chebyshev,
    "linfty": chebyshev,
    "linf": chebyshev,
    "minkowski": minkowski,
    "poincare": poincare,
    # Standardised/weighted distances
    "seuclidean": standardised_euclidean,
    "standardised_euclidean": standardised_euclidean,
    "wminkowski": weighted_minkowski,
    "weighted_minkowski": weighted_minkowski,
    "mahalanobis": mahalanobis,
    # Other distances
    "canberra": canberra,
    "cosine": cosine,
    "correlation": correlation,
    "hellinger": hellinger,
    "haversine": haversine,
    "braycurtis": bray_curtis,
    "ll_dirichlet": ll_dirichlet,
    "symmetric_kl": symmetric_kl,
    # Binary distances
    "hamming": hamming,
    "jaccard": jaccard,
    "dice": dice,
    "matching": matching,
    "kulsinski": kulsinski,
    "rogerstanimoto": rogers_tanimoto,
    "russellrao": russellrao,
    "sokalsneath": sokal_sneath,
    "sokalmichener": sokal_michener,
    "yule": yule,
    # Special discrete distances
    "categorical": categorical_distance,
    "ordinal": ordinal_distance,
    "hierarchical_categorical": hierarchical_categorical_distance,
    "count": count_distance,
    "string": levenshtein,
    "gaussian_energy": gaussian_energy,
    "hyperboloid": hyperboloid,
}

named_distances_with_gradients = {
    # general minkowski distances
    "euclidean": euclidean_grad,
    "l2": euclidean_grad,
    "manhattan": manhattan_grad,
    "taxicab": manhattan_grad,
    "l1": manhattan_grad,
    "chebyshev": chebyshev_grad,
    "linfinity": chebyshev_grad,
    "linfty": chebyshev_grad,
    "linf": chebyshev_grad,
    "minkowski": minkowski_grad,
    # Standardised/weighted distances
    "seuclidean": standardised_euclidean_grad,
    "standardised_euclidean": standardised_euclidean_grad,
    "wminkowski": weighted_minkowski_grad,
    "weighted_minkowski": weighted_minkowski_grad,
    "mahalanobis": mahalanobis_grad,
    # Other distances
    "canberra": canberra_grad,
    "cosine": cosine_grad,
    "correlation": correlation_grad,
    "hellinger": hellinger_grad,
    "haversine": haversine_grad,
    "braycurtis": bray_curtis_grad,
    "symmetric_kl": symmetric_kl_grad,
    # Special embeddings
    "spherical_gaussian_energy": spherical_gaussian_energy_grad,
    "diagonal_gaussian_energy": diagonal_gaussian_energy_grad,
    "gaussian_energy": gaussian_energy_grad,
    "hyperboloid": hyperboloid_grad,
}

DISCRETE_METRICS = (
    "categorical",
    "hierarchical_categorical",
    "ordinal",
    "count",
    "string",
)

SPECIAL_METRICS = (
    "hellinger",
    "ll_dirichlet",
    "symmetric_kl",
    "poincare",
    hellinger,
    ll_dirichlet,
    symmetric_kl,
    poincare,
)


@jax.jit
def parallel_special_metric(X, Y=None, metric=hellinger):
    if Y is None:
        result = jnp.zeros((X.shape[0], X.shape[0]))

        for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                result = result.at[i, j].set(metric(X[i], X[j]))
                result = result.at[j, i].set(result[i, j])
    else:
        result = jnp.zeros((X.shape[0], Y.shape[0]))

        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                result = result.at[i, j].set(metric(X[i], Y[j]))

    return result


# We can gain efficiency by chunking the matrix into blocks;
# this keeps data vectors in cache better
@jax.jit
def chunked_parallel_special_metric(X, Y=None, metric=hellinger, chunk_size=16):
    if Y is None:
        XX, symmetrical = X, True
        row_size = col_size = X.shape[0]
    else:
        XX, symmetrical = Y, False
        row_size, col_size = X.shape[0], Y.shape[0]

    result = jnp.zeros((row_size, col_size), dtype=jnp.float32)
    n_row_chunks = (row_size // chunk_size) + 1
    for chunk_idx in range(n_row_chunks):
        n = chunk_idx * chunk_size
        chunk_end_n = min(n + chunk_size, row_size)
        m_start = n if symmetrical else 0
        for m in range(m_start, col_size, chunk_size):
            chunk_end_m = min(m + chunk_size, col_size)
            for i in range(n, chunk_end_n):
                for j in range(m, chunk_end_m):
                    result = result.at[i, j].set(metric(X[i], XX[j]))
    return result


def pairwise_special_metric(
        X, Y=None, metric="hellinger", kwds=None, ensure_all_finite=True
):
    if callable(metric):
        if kwds is not None:
            kwd_vals = tuple(kwds.values())
        else:
            kwd_vals = ()

        @jax.jit
        def _partial_metric(_X, _Y=None):
            return metric(_X, _Y, *kwd_vals)

        return pairwise_distances(
            X, Y, metric=_partial_metric, ensure_all_finite=ensure_all_finite
        )
    else:
        special_metric_func = named_distances[metric]
    return parallel_special_metric(X, Y, metric=special_metric_func)
