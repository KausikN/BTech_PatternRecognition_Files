"""
Edit Distance
"""

# Imports
import numpy as np

# Driver Code
def EditDistance(S1, S2):
    dist = 0

    S1 = list(S1)
    S2 = list(S2)

    dist = max(len(S1), len(S2)) - min(len(S1), len(S2))

    for i in range(min(len(S1), len(S2))):
        if not S1[i] == S2[i]:
            dist += 1

    return dist

# Driver Code
# Params
S1 = 'HELLO'
S2 = 'HELLO'
# Params

# RunCode
dist = EditDistance(S1, S2)
print("S1:\n", S1)
print("S2:\n", S2)
print("Edit Distance:\n", dist)