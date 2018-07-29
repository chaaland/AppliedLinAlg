
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

    squared_diff = (x - y).^2;
    mean_square_error = mean(squared_diff);

    return sqrt(mean_square_error)
end
