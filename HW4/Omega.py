


def Omega(er, returns, rf, target=0):
	return (er - rf) / lpm(returns, target, 1)

def omega_ratio(er, returns, rf, target=0):
    return (er - rf) / lpm(returns, target, 1)


def lpm(returns, threshold, order):
    # This method returns a lower partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = numpy.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the threshold and the returns
    diff = threshold_array - returns
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return numpy.sum(diff ** order) / len(returns)

