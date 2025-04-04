import numpy as np
np.random.seed(321)

N = 1000
prob_of_coming, prob_of_voting_Alice = np.random.rand(N), np.random.rand(N)

chances_Alice = []
for _ in range(10000):
    came_voting = np.random.binomial(n=1, p=prob_of_coming)
    want_to_vote_Alice = np.random.binomial(n=1, p=prob_of_voting_Alice)

    vote_A = np.sum(came_voting & want_to_vote_Alice)
    vote_B = np.sum(came_voting & 1 - want_to_vote_Alice)

    chances_Alice.append(vote_A > vote_B)

print(f'The probability of Alice to win is ± {np.average(chances_Alice)}')


