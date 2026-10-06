# Negative beta is in km^-1; si_to_od supplies Euclidean distance in metres.
# Origin-production normalisation is performed by simodels::si_calculate.
gravity_model = function(beta, d, m, n) {
  if (length(beta) != 1L || !is.finite(beta) || beta >= 0) {
    stop("beta must be one finite negative coefficient in inverse kilometres")
  }
  if (any(!is.finite(d)) || any(d < 0) ||
      any(!is.finite(m)) || any(m < 0) ||
      any(!is.finite(n)) || any(n < 0)) {
    stop("distance, production and attractiveness must be finite and nonnegative")
  }
  m * n * exp(beta * d / 1000)
}
