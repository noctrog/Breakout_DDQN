import numpy as np

class SumTree:
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity

        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def __len__(self):
        return self.capacity

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data
        self.update(idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p

        self._propagate(idx, change)

    def get(self, s):
        parent_idx = 0
        leaf_idx = 0

        while True:
            left = 2 * parent_idx + 1
            right = left + 1

            if left >= len(self.tree):
                leaf_idx = parent_idx
                break

            if s <= self.tree[left]:
                parent_idx = left
            else:
                s = s - self.tree[left]
                parent_idx = right

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

class PrioritizedReplayBuffer(object):
    PRB_alpha = 0.6
    PRB_beta = 0.4
    PRB_beta_increment = 0.001
    PRB_max_error = 1.0
    PRB_e = 0.01

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.empty = True

    def __len__(self):
        return len(self.tree)

    def _get_priority(self, error):
        return (error + self.PRB_e) ** self.PRB_alpha

    def max_priority(self):
        return np.max(self.tree.tree[-self.tree.capacity:])

    def add(self, error, experience):
        self.empty = False

        priority = self._get_priority(error)
        self.tree.add(priority, experience)

    def sample(self, n):
        batch = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n,1), dtype=np.float32)

        priority_segment = self.tree.total() / n

        self.PRB_beta = np.min([1., self.PRB_beta + self.PRB_beta_increment])

        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()
        max_weight = (p_min * n) ** (-self.PRB_beta)

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get(value)

            sampling_probabilities = priority / self.tree.total()

            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PRB_beta) / max_weight
            b_idx[i] = index

            experience = data
            batch.append(experience)

        return b_idx, batch, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PRB_e
        clipped_errors = np.minimum(abs_errors, self.PRB_max_error)
        ps = np.power(clipped_errors, self.PRB_alpha)

        for tim, p in zip(tree_idx, ps):
            self.tree.update(tim, p)
