"""
    This is a test to test the "growing Qs bug"
"""


def get_new_q_value(q_old):
    delta_sum = 0
    for i in range(100):
        # i is index in trajectory
        q_new = 0
        for j in range(100-i):
            rewards = 100 - i - j
            if i + j != 99:
                rewards += q_old
            q_new += rewards
        q_new = q_new/(100-i)
        delta_sum += q_new - q_old
    return delta_sum

q = 1000
while True:
    q += 0.1 * get_new_q_value(q)
    print(q)