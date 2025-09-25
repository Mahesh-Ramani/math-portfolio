Check out factorials.pdf on information about n!.py.

n!.py outputs the approximate value of n! based on a given value of n.

I present an algorithm that computes n! for large n by using prime factor grouping and a balanced product tree to reduce multiplication work, achieving asymptotic cost improvements over naive multiplication in practice (bordering on the order of thousands of times better than naiive methods for n>10**6)
