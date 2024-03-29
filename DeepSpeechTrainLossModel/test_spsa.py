from scipy.optimize import minimize

def f(x):
	return x[0]**2 + x[1]**2
	
	
x = [1,2]
ans = minimize(f, [1,2], method='Nelder-Mead')
print(ans)
