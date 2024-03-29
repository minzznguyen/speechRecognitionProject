import numpy as np
import operator
from math import sqrt
def spsa(f, x, adam=False, iterations=1000, lr=None, lr_decay=0.9, lr_power=0.25,
         px=0.01, px_decay=0.9, px_power=0.25, momentum=0.9, beta=0.9, epsilon=1e-8):
    x = np.array(x, dtype=float)
    adam = bool(operator.index(adam))
    iterations = operator.index(iterations)
    if lr is not None:
        lr = float(lr)
    lr_decay = float(lr_decay)
    lr_power = float(lr_power)
    if px is int:
        x_temp = np.empty_like(x, dtype=int)
    elif px is not None:
        px = float(px)
    px_decay = float(px_decay)
    px_power = float(px_power)
    momentum = float(momentum)
    beta = float(beta)
    epsilon = float(epsilon)
    rng = np.random.default_rng()

    # Rest of the code...
    m1 = 1.0 - momentum
    m2 = 1.0 - beta
    # Estimate the noise in f.
    bn = 0.0
    y = 0.0
    noise = 0.0
    for _ in range(int(sqrt(x.size + 100))):
        temp = f(x)
        bn += m2 * (1 - bn)
        y += m2 * (temp - y)
        noise += m2 * ((temp - f(x)) ** 2 - noise)
    # Estimate the gradient and its square.
    b1 = 0.0
    b2 = 0.0
    gx = np.zeros_like(x)
    slow_gx = np.zeros_like(x)
    square_gx = np.zeros_like(x)
    for i in range(int(sqrt(x.size + 100))):
        # Compute df/dx in random directions.
        if px is int:
            dx = rng.choice((-0.5, 0.5), x.shape)
            df_dx = (f(np.rint(x + dx, casting="unsafe", out=x_temp)) - f(np.rint(x - dx, casting="unsafe", out=x_temp))) * 0.5 / dx
        else:
            dx = rng.choice((-1.0, 1.0), x.shape) / (1 + i)
            df_dx = (f(x + dx) - f(x - dx)) * 0.5 / dx
        # Update the gradients.
        b1 += m1 * (1 - b1)
        b2 += m2 * (1 - b2)
        gx += m1 * (df_dx - gx)
        slow_gx += m2 * (df_dx - slow_gx)
        square_gx += m2 * ((slow_gx / b2) ** 2 - square_gx)
    print(df_dx, gx)
    # Estimate the learning rate.
    if lr is None:
        lr = 1e-5
        # Increase the learning rate while it is safe to do so.
        dx = 3 / b1 * gx
        if adam:
            dx /= np.sqrt(square_gx / b2 + epsilon)
        for _ in range(3):
            while f(x - lr * dx) < f(x):
                lr *= 1.4
    # Track the average value of x.
    mx = sqrt(m1 * m2)
    bx = mx
    x_avg = mx * x
    # Track the best (x, y).
    y_best = y / bn
    x_best = x.copy()
    # Track how many times the solution fails to improve.
    momentum_fails = 0
    consecutive_fails = 0
    improvement_fails = 0
    # Initial step size.
    dx = gx / b1
    
    if adam:
        dx /= np.sqrt(square_gx / b2 + epsilon)
    # Run the number of iterations.
    for i in range(iterations):
        # Estimate the next point.
        x_next = x - lr * dx
        # Compute df/dx in at the next point.
        if px is int:
            dx = rng.choice((-0.5, 0.5), x.shape)
            y1 = f(np.rint(x_next + dx, casting="unsafe", out=x_temp))
            y2 = f(np.rint(x_next - dx, casting="unsafe", out=x_temp))
        else:
            dx = (lr / m1 * px / (1 + px_decay * i) ** px_power) * np.linalg.norm(dx)
            if adam:
                dx /= np.sqrt(square_gx / b2 + epsilon)
            dx *= rng.choice((-1.0, 1.0), x.shape)
            y1 = f(x_next + dx)
            y2 = f(x_next - dx)
        df = (y1 - y2) / 2
        df_dx = dx * (df * sqrt(x.size) / np.linalg.norm(dx) ** 2)
        # Update the momentum.
        if (df_dx.flatten() / np.linalg.norm(df_dx)) @ (gx.flatten() / np.linalg.norm(gx)) < 0.5 / (1 + 0.1 * momentum_fails) ** 0.3 - 1:
            momentum_fails += 1
            m1 = (1.0 - momentum) / sqrt(1 + 0.1 * momentum_fails)
        # Update the gradients.
        b1 += m1 * (1 - b1)
        b2 += m2 * (1 - b2)
        gx += m1 * (df_dx - gx)
        slow_gx += m2 * (df_dx - slow_gx)
        square_gx += m2 * ((slow_gx / b2) ** 2 - square_gx)
        # Compute the step size.
        dx = gx / (b1 * (1 + lr_decay * i) ** lr_power)
        if adam:
            dx /= np.sqrt(square_gx / b2 + epsilon)
        # Sample points.
        y3 = f(x)
        y4 = f(x - lr * 0.5 * dx)
        y5 = f(x - lr / sqrt(m1) * dx)
        y6 = f(x)
        # Estimate the noise in f.
        bn += m2 * (1 - bn)
        y += m2 * (y3 - y)
        noise += m2 * ((y3 - y6) ** 2 + 1e-64 * (abs(y3) + abs(y6)) - noise)
        # Perform line search.
        # Adjust the learning rate towards learning rates which give good results.
        if y3 - 0.25 * sqrt(noise / bn) < min(y4, y5):
            lr /= 1.3
        if y4 - 0.25 * sqrt(noise / bn) < min(y3, y5):
            lr *= 1.3 / 1.4
        if y5 - 0.25 * sqrt(noise / bn) < min(y3, y4):
            lr *= 1.4
        # Set a minimum learning rate.
        lr = max(lr, epsilon / (1 + 0.01 * i) ** 0.5 * (1 + 0.25 * np.linalg.norm(x)))
        # Update the solution.
        x -= lr * dx
        bx += mx / (1 + 0.01 * i) ** 0.303 * (1 - bx)
        x_avg += mx / (1 + 0.01 * i) ** 0.303 * (x - x_avg)
        consecutive_fails += 1
        # Track the best (x, y).
        if y / bn < y_best:
            y_best = y / bn
            x_best = x_avg / bx
            consecutive_fails = 0
        if consecutive_fails < 128 * (improvement_fails + int(sqrt(x.size + 100))):
            continue
        # Reset variables if diverging.
        consecutive_fails = 0
        improvement_fails += 1
        x = x_best
        bx = mx * (1 - mx)
        x_avg = bx * x
        noise *= m2 * (1 - m2) / bn
        y = m2 * (1 - m2) * y_best
        bn = m2 * (1 - m2)
        b1 = m1 * (1 - m1)
        gx = b1 / b2 * slow_gx
        slow_gx *= m2 * (1 - m2) / b2
        square_gx *= m2 * (1 - m2) / b2
        b2 = m2 * (1 - m2)
        lr /= 64 * improvement_fails
    if px is int:
        x_best = np.rint(x_best).astype(int)
        x = np.rint(x, casting="unsafe", out=x_temp)

    # Implement the remaining code as given in the original function.

    return x_best if y_best - 0.25 * sqrt(noise / bn) < min(f(x), f(x)) else x

