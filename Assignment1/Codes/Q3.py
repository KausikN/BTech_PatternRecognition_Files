'''
Assignment Q3
Q3. Compare two text files doc1.txt and doc2.txt using cosine distance
'''

# Imports
import numpy as np

# Main Functions
def ReadFile(path):
    return open(path, 'r').read()

def CosineDistance(P1, P2):
    dist = 0

    # Convert both to same length
    if not len(P1) == len(P2):
        if len(P1) > len(P2):
            P2 = P2 + [0]*(len(P1)-len(P2))
        else:
            P1 = P1 + [0]*(len(P2)-len(P1))

    u = np.array(P1)
    v = np.array(P2)

    dist = np.dot(u, v) / ((np.dot(u, u)**(0.5)) * (np.dot(v, v)**(0.5)))

    return dist

# Driver Code
# Params
doc1_Path = 'Assignment1/Data/doc1.txt'
doc2_Path = 'Assignment1/Data/doc2.txt'
# Params

# RunCode
doc1_Text = ReadFile(doc1_Path)
doc2_Text = ReadFile(doc2_Path)

print("")
print("Doc 1 (" + str(len(doc1_Text)) + " chars, " + str(len(doc1_Text.split(' '))) + " words, " + str(len(doc1_Text.split('.'))-1) + " lines):\n", doc1_Text)
print("Doc 2 (" + str(len(doc2_Text)) + " chars, " + str(len(doc2_Text.split(' '))) + " words, " + str(len(doc2_Text.split('.'))-1) + " lines):\n", doc2_Text)
print("")

# Cosine Distance
# Convert Text to ASCII list
P1 = list(map(ord, doc1_Text))
P2 = list(map(ord, doc2_Text))

# Compute Cosine Distance
CosineDist = CosineDistance(P1, P2)
print("Cosine Distance:", CosineDist)