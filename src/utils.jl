using LinearAlgebra;
# using Statistics;

function rotate_mat2d(angle::T; ccw=true) where T <: Real
    #= Helper for rotating points in 2d space
    
    Builds the rotation matrix specified by the angle and the direction of rotation
    
    Args :
        angle : angle in radians of rotation
        ccw : whether the angle is measured counter clockwise wrt the positive x axis
    
    Returns :
        A 2x2 rotation matrix
    =#

    rotate_mat = zeros(2,2);
    if ccw
        rotate_mat = [cos(angle) -sin(angle);
                      sin(angle) cos(angle)];
    else
        rotate_mat = [cos(angle) sin(angle);
                      -sin(angle) cos(angle)];
    end
    
    return rotate_mat;
end

function vandermonde(x, n)
    #= Generates a vandermonde matrix using the entries of x

    Given a vector x, the successive powers of x up to and including n
    are computed and stored as rows of a matrix

    Args :
        x : a vector of values
        n : the degree of the polynomial

    Returns :
        A matrix of dimension length(x) by n+1 where a row is given by
            [1 a a^2 ... a^{n-1} a^n]
    =#

    A = zeros(length(x), n + 1);
    xtilde = vec(x);
    col = ones(size(xtilde));
    for i = 0:n
        A[:,i+1] = col;
        col = col .* xtilde;
    end

    return A
end

function rmse(x; y=0)
    #= Computes the root mean square error between x and y 

    Given two vectors x and y, return
        sqrt(1/N * (sum((x_i - y_i)^2)))
    =#

    squared_diff = (x .- y).^2;
    mean_square_error = sum(squared_diff) / length(squared_diff)

    return sqrt(mean_square_error)
end


function net_present_value(c::Array{T,1}, r::Real) where T <: Real
    n = length(c);
    return sum(c .* (1 + r).^(-(0:n-1)))
end

function net_present_value_grad(c::Array{T,1}, r::Real) where T<:Real
    n = length(c);
    return sum(-(0:n-1) .* c .* (1 + r).^(-(1:n)))
end

function state_propagation_matrix(A::Array{T,2}, B::Array{T,2}, n::Real) where T <: Real
    #= Returns state propagation matrix of a linear dynamical system

    Given a discrete time linear dynamical system of the form 

                x[n] = A * x[n] + B * u[n]

    we can expand the recursion to get x[n] as a function of the initial 
    state and the inputs at each time

            x[n] = A^{n-1} * x[1] + A^{n-2} * B * u[1] + ... + A^2B * u[n-3] + AB * u[n-2] + B * u[n-1]
        
    The matrix multiplying [u[1]; u[2]; u[3];...; u[n-1]] is the matrix 
    returned by this function. In the case where n is the size of the 
    state, this the controllability matrix of the system.

    Args :
        A : The state evolution matrix. Must be square
        B : Input to Output matrix
        n : Indexed from 1, the desired state needed to compute
    
    Returns :
        The matrix A^{n-1} needed for propagating the initial state and
        the matrix [A^{n-2}B A^{n-3}B ... A^2B AB B] for propagating the
        inputes

    =#

    if size(A,1) != size(A,2)
        error("Matrix 'A' must be square");
    end
    if size(A,1) != size(B,1)
        error("Matrix 'A' and 'B' must have same row num")
    end

    num_states = size(A,1);
    num_inputs = size(B, 2);

    C = zeros(num_states, (n - 1) * num_inputs);

    Apow = I

    for i in n-1:-1:1
        left = (i - 1) * num_inputs + 1;
        right = num_inputs * i;
        C[1:end, left:right] = Apow * B;
        Apow *= A;
    end

    return  Apow, C
end

function propagate_linear_dyanmical_system(A::Array{T,2}, B::Array{T,2}, U::Array{T,2}, xinit::Array{T,1}) where T <: Real
    #= Given the initial state, the inputs and the matrices for an LDS compute the trajectory
    
    Simulate the LDS given the inputs, inital state, the input-output matrix and the state 
    transitino matrix. The discrete time linear dynamical system has recurrence

                    x[k] = A * x[k-1] + B * u[k-1]

    Args :
        A : The n x n state evolution matrix
        B : Input to Output matrix of size m x n
        U : Inputs matrix of size m x (T - 1) where each input is a column
        xinit : Initial state of the system
    
    Returns :
        X : states of size n x T where each column is a state at time point k
    =#

    state_dim = size(A, 1);
    input_dim = size(U, 1);
    num_inputs = size(U, 2);

    xinit = vec(xinit);
    X = zeros(state_dim, num_inputs + 1);

    X[:,1] = xinit;
    for i in 1:num_inputs
        X[:, i+1] = A * states[:,i] + B * U[:,i];
    end
     
    return X
end

function propagate_linear_dyanmical_system_with_state_feedback(A::Array{T,2}, B::Array{T,2}, K::Array{T,2}, tfinal::N, 
                                                          xinit::Array{T,1}) where T <: Real where N <:Int64
    #= Given the initial state, the state feedback matrix and the matrices for an LDS compute the trajectory
    
    Simulate the LDS using state feedback control. The discrete time linear dynamical 
    system with state feedback control is given by 

                    x[k] = (A + B * K) * x[k-1] 

    Args :
        A : The n x n state evolution matrix
        B : Input to Output matrix of size m x n
        K : time invariant state feedback control matrix
        xinit : Initial state of the system
    
    Returns :
        X : states matrix of size n x T where each column is a state at time point k
        U : inputs of state feedback controller given as m x (T - 1)
    =#

    state_dim = size(A, 1);
    input_dim = size(B, 2);

    X = zeros(state_dim, tfinal);
    U = zeros(input_dim, tfinal - 1);
    X[:,1] = xinit;
   
    for i in 2:tfinal
        U[:,i-1] = K * X[:,i-1];
        X[:,i] = A * X[:,i-1] + B * U[:,i-1];
    end

    return X, U;
end

function lqr_matrix(A::Array{T,2}, B::Array{T,2}, C::Array{T,2}, xinit::Vector{T}, tfinal::N, rho::Real) where T <: Real where N <: Int64
    #= Creates the large block matrix needed for linear quadratic control

    The linear quadratic control problem can be expressed as a least squares problem

            minimize    ||Atilde * x - btilde||^2
            subject to  Ctilde * x = dtilde

    This function returns the matrices needed for the problem formulation

    Args :
        A : The n x n state evolution matrix
        B : Input to Output matrix of size m x n 
        C : Input to Observation matrix of size p x n
        xinit : initial state of the system as an n x 1 vector
        tfinal : determines the number of inputs to choose for the linear quadratic control problem
        rho : weighting of the input norm in the least squares objective trading off state size with input size

    Returns :
        The matrices needed to formulate the constrained least squares problem for linear quadratic control
        
    =#
    n = size(A, 1);
    m = size(B, 2);
    
    upLeft = Matrix{Float64}(I, tfinal * n, tfinal * n);
    for i in 1:tfinal
        lowerInd = n * (i - 1) + 1;
        upperInd = lowerInd + n - 1;
        upLeft[lowerInd:upperInd, lowerInd:upperInd] = C;
    end

    lowRight = Matrix{Float64}(sqrt(rho) * I, (tfinal - 1) * m, (tfinal - 1) * m);
    Atilde = [upLeft zeros(tfinal * n, (tfinal - 1) * m); 
              zeros((tfinal - 1) * m, tfinal * n) lowRight];
    btilde = zeros(size(Atilde, 1));
    
    upperLeft = zeros((tfinal - 1) * n, n * tfinal);
    subMat = [A  Matrix{Float64}(-I, n, n)];

    for i in 1:(tfinal - 1)
        lowerInd = n * (i - 1) + 1;
        upperColInd = lowerInd + 2 * n - 1;
        upperRowInd = lowerInd + n - 1;
        upperLeft[lowerInd:upperRowInd, lowerInd:upperColInd] = subMat;
    end

    upperRight = zeros(n * (tfinal - 1), m * (tfinal - 1));
    for i in 1:(tfinal - 1)
        lowerRowInd = (i - 1) * n + 1;
        lowerColInd = (i - 1) * m + 1;
        upperRowInd = i * n;
        upperColInd = i * m;
        upperRight[lowerRowInd:upperRowInd,lowerColInd:upperColInd] = B;
    end

    lowerLeft = [Matrix{Float64}(I, n, n) zeros(n, n * (tfinal - 1))];

    Ctilde = [upperLeft upperRight; 
             lowerLeft zeros(n, m * (tfinal - 1))];
    dtilde = zeros(size(Ctilde,1));
    dtilde[end-n+1:end,1] = xinit;
    
    return Atilde, vec(btilde), Ctilde, vec(dtilde)
end

function constrained_least_squares(A::Array{T,2}, b::Vector{T}, C::Array{T,2}, d::Vector{T}) where T <: Real
    #= Solves the constrained least squares problem

    Given a least squares problem of the form 
        minimize    ||Ax - b||^2
        subject to  Cx = d
    
    Args :
        A : m x n matrix
        b : m x 1 vector
        C : p x n matrix
        d : p x 1 vector
    
    Returns :
        xstar : solution of the constrained least squares problem
    =#

    k = size(C,1)
    kktMatrix = [2*A'*A C'; 
                C zeros(k, k)];
    kktRHS = [2*A'*b; d];
    xstar = kktMatrix \ kktRHS;

    return xstar
end

function levenberg_marquardt(input_output_shape::Tuple{Int64,Int64}, f::Function, J::Function; xinit=Inf, max_iters=1000, atol=1e-6)
    #= Implements the levenberg marquardt heuristic for finding roots of m nonlinear equations in n unknowns
    
    Args :
        input_output_shape : a tuple of the form (n, m) giving the number of variables
                             and the number of outputs respectively
        f : a function that takes 'input_dim' inputs and returns m outputs
        J : a function to evaluate the Jacobian of 'f'
        xinit : initial iterate for warm starting
        max_iters : the maximum number of iterations to perform 
        atol : the absolute tolerance of the root mean square of the Jacobian

    Returns :
        xvals : the trajectory of the gradient descent
        fvals : the value of the objective along the trajectory
        stop_criteria : root mean square of twice the transposed jacobian times the evaluation of the function
        lambdavals : the values of the penalty parameter for each iteration of the algo

    =#
    LAMBDA_MAXIMUM = 1e10;
    STEPSZ_MINIMUM = 1e-5;

    n = input_output_shape[1];
    m = input_output_shape[2];

    if any(isinf.(xinit))                 
        xinit = vec(randn(n));
    end
    
    lambdavals = [1];
    xvals = hcat(xinit);
    xcurr = vec(xvals);
    fvals = vcat(f(xcurr));

    total_deriv = J(xcurr);
    stop_criteria = rmse(2 * total_deriv' * f(xcurr));

    for i in 1:max_iters
        while true
            if m == 1
                A = vcat(total_deriv, sqrt(lambdavals[i]) * Matrix{Float64}(I, n, n));
                # A = vcat(total_deriv, sqrt(lambdavals[i]) * eye(n,n));
                b = vcat(total_deriv .* xvals[:,i] .- fvals[:,i], sqrt(lambdavals[i]) * xvals[:,i]);
                xcurr = A \ b;
            else
                A = vcat(total_deriv, sqrt(lambdavals[i]) * Matrix{Float64}(I, n, n));
                #A = vcat(total_deriv, sqrt(lambdavals[i]) * eye(n,n));
                b = vcat(total_deriv * xvals[:,i] - fvals[:,i], sqrt(lambdavals[i]) * xvals[:,i]);
                xcurr = A \ b;
            end

            if norm(f(xcurr)) < norm(fvals[:,i])
                lambdavals = hcat(lambdavals, lambdavals[i] * 0.8);
                break
            elseif 2 * lambdavals[i] > LAMBDA_MAXIMUM
                lambdavals = hcat(lambdavals, lambdavals[i]);
                break;
            elseif rmse(xcurr - xvals[:,i]) < STEPSZ_MINIMUM
                lambdavals = hcat(lambdavals, lambdavals[i]);
                break;
            else
                lambdavals[i] = 2 * lambdavals[i];
            end
        end

        xvals = hcat(xvals, xcurr);
        fvals = hcat(fvals, f(xcurr));
        total_deriv = J(xcurr); 
        stop_criteria = hcat(stop_criteria, rmse(2 * total_deriv' * f(xcurr)));
        if stop_criteria[end] <= atol                   # From grad ||f(x)||^2
            break
        end
    end

    return xvals, fvals, vec(stop_criteria), vec(lambdavals)
end

function augmented_lagrangian(input_output_shape::Tuple{Int64,Int64,Int64}, f::Function, J1::Function, g::Function, J2::Function; xinit=Inf, max_iters=100, atol=1e-6)
    #= Solves the constrained non-linear least squares by augmenting the lagrangian with the penalty objective.

    Args :
    Returns :
     
    =#
    n = input_output_shape[1];
    m = input_output_shape[2];
    p = input_output_shape[3];

    if any(isinf.(xinit))                 
        xinit = vec(randn(n));
    end

    mu = hcat(1);
    xtraj = hcat(xinit);
    ztraj = vec(zeros(p));
    cume_iters = hcat(0);

    for i = 1:max_iters
        sqrtmu = sqrt(mu[i])
        xvals, fvals, gradnorm, lambdavals = levenberg_marquardt((n, m + p)
                                                                ,x -> vcat(f(x), sqrtmu * g(x) + 0.5 * ztraj[:,i]/sqrtmu) 
                                                                ,x -> vcat(J1(x), sqrt(mu[i]) * J2(x))
                                                                ,xinit=xtraj[:,i]);
        xtraj = hcat(xtraj, xvals[:,end]);
        cume_iters = hcat(cume_iters, cume_iters[i] + size(xvals,2));
        ztraj = hcat(ztraj, ztraj[:,i] + 2 * mu[i] * g(xtraj[:,i+1]));

        if norm(g(xtraj[:,i+1])) >= 0.25 * norm(g(xtraj[:,i]))
            mu = hcat(mu, 2 * mu[i]);
        else
            mu = hcat(mu, mu[i]);
        end

        if norm(g(xtraj[:,i+1])) <= atol 
            return xtraj, vec(mu), vec(cume_iters);
        end
    end

    return xtraj, vec(mu), vec(cume_iters)
end

function penalty_algo(input_output_shape::Tuple{Int64,Int64,Int64}, f::Function, J1::Function, g::Function, J2::Function; xinit=Inf, max_iters=1000, atol=1e-6)
    #= Naive penalty algo for non-linear equality constrained optimization
    
    A method for solving the minimization of ||f(x)||^2 subject to g(x) = 0 
    where g : R^n -> R^m and potentially non linear. The method solves a 
    sequence of minimizations of the form 

                ||f(x)||^2 + mu * ||g(x)||^2

    as a heuristic (where mu is steadily increased each iteration).

    Args :
        input_output_shape : a tuple of the form (n, m) giving the number of variables
                             and the number of outputs respectively
        f : a function that takes 'n' inputs and returns m outputs
        J1 : a function to evaluate the Jacobian of 'f'
        g : a function that takes 'n' inputs and returns p outputs
        J2 : a functtion to evaluate the Jacobian of 'g'
        xinit : an initial point used for warm starting the algo
        max_iters : maximum number of iterations 
        atol : tolerance of the root mean square error of the equality condition
    
    Returns :
        xtraj : the sequence of iterates in the minimization
        mu : the sequence of penalties
    =#

    n = input_output_shape[1];
    m = input_output_shape[2];
    p = input_output_shape[3];

    if any(isinf.(xinit))                 
        xinit = vec(randn(n));
    end

    mu = hcat(1);
    xtraj = hcat(xinit);
    cume_iters = hcat(0);
    for i = 1:max_iters
        sqrtmu = sqrt(mu[i]);
        objective = vcat(f(xtraj[:,i]), sqrtmu * g(xtraj[:,i]));
        deriv = vcat(J1(xtraj[:,i]), sqrtmu * J2(xtraj[:,i]));

        xvals, fvals, gradnorm, lambdavals = levenberg_marquardt((n, m + p)
                                                                ,x -> vcat(f(x), sqrtmu * g(x)) 
                                                                ,x -> vcat(J1(x), sqrtmu * J2(x))
                                                                ,xinit=xtraj[:,i]);

        xtraj = hcat(xtraj, xvals[:,end]);
        mu = hcat(mu, 2*mu[i]);
        cume_iters = hcat(cume_iters, cume_iters[i] + size(xvals,2));

        if rmse(g(xtraj[:,end])) <= atol 
            return xtraj, vec(mu), vec(cume_iters)
        end
    end

    return xtraj, vec(mu), vec(cume_iters)
end

function gauss_newton(input_output_shape::Tuple{Int64,Int64}, f::Function, J::Function; xinit=Inf, max_iters=1000, atol=1e-6)
    #= Use the gauss-newton method to find extrema
    
    The Gauss-Newton method is used to approximately solve the non-linear least
    squares problem (NNLS). The method employs only first order information to 
    locate optima

    Args :
        input_dim : dimension of the input to the function to be minimized
        f : a function representing the residuals of the m nonlinear equations
        J : a function to evaluate the jacobian of 'f' 
        xinit : initial iterate for warm starting
        max_iters : the maximum number of newton steps to take
        atol : the absolute tolerance of the root mean square of the jacobian

    Returns :
        xvals : the trajectory of the gradient descent
        fvals : the value of the objective along the trajectory
        stop_criteria : the norm of the jacobian along the trajectory

    =#

    n = input_output_shape[1];
    m = input_output_shape[2];

    if any(isinf.(xinit))                 
        xinit = vec(randn(n));
    end
    
    xvals = hcat(xinit);
    xcurr = vec(xvals);
    fvals = vcat(f(xcurr));

    total_deriv = J(xcurr);
    stop_criteria = rmse(2*total_deriv' * f(xcurr));

    for i=1:max_iters
        if m == 1
            A = vcat(total_deriv);
            b = vcat(total_deriv .* xvals[:,i] .- fvals[:,i]);
            xcurr = A \ b;
        else
            A = vcat(total_deriv);
            b = vcat(total_deriv * xvals[:,i] - fvals[:,i]);
            xcurr = A \ b;
        end

        xvals = hcat(xvals, xcurr);
        fvals = hcat(fvals, f(xcurr));
        total_deriv = J(xcurr);
        stop_criteria = hcat(stop_criteria, rmse(2*total_deriv' * f(xcurr)));
        if rmse(2*total_deriv' * f(xcurr)) <= atol                   # From grad ||f(x)||^2
            break
        end
    end
    
    return xvals, fvals, vec(stop_criteria)
end

function parametric2ellipse_coords(semiaxis_lengths::Array{T,1}; center=[0 0], ccw_angle=0, numpoints=1000) where T <: Real
    #= Helper for plotting a (possibly degenerate) 2D ellipse given semi-major/minor
    axes lengths, ellipse center coordinates and angle off the positive x axis
    
    Given the positve semidefinite matrix of a quadratic form specifying an ellipse
    in standard form x^T A x = 1, the center of the ellipse, the angle to rotate the
    ellipse wrt the positive x-axis, and the number of points desired for plotting, 
    an array containing the x, y coordinates of various points on the ellipse are returned
    
    Args :
        semiaxis_lengths : array of the form [a b] where a is half the length of the axis aligned
                      ellipse along the x-axis and b is half the length along the y-axis (before
                      rotation)
        center : array of the x and y coordinates of the center of the ellipse
        ccw_angle : The counter clockwise angle (in rad) to rotate the ellipse wrt
                    the positive x-axis
        num_points : an integer indicating the number of xy pairs on the ellipse to return
    
    Returns :
        A 2 x numpoints array where the x and y coordinates are in the 1st and 2nd row 
        respectively
    =#

    if size(vec(center))[1] != 2
        error("Parameter 'center' must be of size 2");
    end

    if size(vec(semiaxis_lengths))[1] != 2
        error("Parameter 'semiaxis_lengths' must be of size 2");
    end
    center = vec(center);
    semiaxis_lengths = vec(semiaxis_lengths);
    
    theta = vec(range(0, stop=2*pi, length=numpoints));
    onaxis_ellipse = semiaxis_lengths .* [cos.(theta) sin.(theta)]';
    
    return center .+ rotate_mat2d(ccw_angle) * onaxis_ellipse;
end
