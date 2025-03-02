def _qdwh(x, is_symmetric, max_iterations):
    eps = float32.machine_eps
    alpha = sqrt(norm(x, 1) * norm(x, inf))
    l = eps

    u = x / alpha

    tol_l = 10.0 * eps / 2.0
    tol_norm = cbrt(tol_l)

    iter_idx = 1
    is_unconverged = True
    is_not_max_iteration = True

    '''
    state = tuple w/:
    - u: current estimate of matrix U in polar decomp
    - l: scalar that tracks the progress of iteration
    - iter_idx: current iteration index
    - is_unconverged
    - is_not_max_iteration
    '''

    while is_unconverged and is_not_max_iteration:
        u_prev = u

        # Compute params
        l2 = l**2
        dd = cbrt(4.0 * (1.0 / l2 - 1.0) / l2)
        sqd = sqrt(1.0 + dd)
        a = (sqd + sqrt(8.0 - 4.0 * dd + 8.0 * (2.0 - l2) / (l2 * sqd)) / 2)
        a = real(a)
        b = (a - 1.0)**2 / 4.0
        c = a + b - 1.0

        # Update l
        l = l * (a + b * l2) / (1.0 + c * l2)

        if c > 100:
            u = qr(u)
        else:
            u = cholesky(u)
        
        if is_symmetric:
            u = (u + u.T.conj()) / 2.0
        
        # checks convergence
        iterating_l = abs(1.0 - l) > tol_l
        iterating_u = norm((u-u_prev)) > tol_norm
        is_unconverged = iterating_l or iterating_u

        is_not_max_iteration = iter_idx < max_iterations

        iter_idx += 1

    # Apply Newton-Schulz refinement for better accuracy
    u = 1.5 * u - 0.5 * u @ (u.T.conj() @ u)

    # Final computation of h
    h = u.T.conj() @ x
    h = (h + h.T.conj()) / 2.0

    # Check if the loop converged
    is_converged = not is_unconverged
    return u, h, iter_idx - 1, is_converged