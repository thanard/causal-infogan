import numpy as np
import networkx as nx
from astar import AStar

'''
to install astar:
git clone https://github.com/jrialland/python-astar.git
python3 setup.py install
'''


def discretize(x, discretization_bins=20, unif_range=(-1, 1)):
    try:
        assert type(x) == np.ndarray
        assert (unif_range[0] < x).all() and (x < unif_range[1]).all()
    except AssertionError:
        import ipdb
        ipdb.set_trace()
    bins = np.linspace(unif_range[0], unif_range[1], num=discretization_bins)
    return np.digitize(x, bins)


def undiscretize(x, discretization_bins=20, unif_range=(-1, 1)):
    try:
        assert type(x) == np.ndarray
        assert (0 < x).all() and (x < discretization_bins).all()
    except AssertionError:
        import ipdb
        ipdb.set_trace()
    bins = np.linspace(unif_range[0], unif_range[1], num=discretization_bins)
    return 0.5 * (bins[x] + bins[x - 1])


class StateObsTuple():
    def __init__(self, state, obs):
        self.state = state.astype(int)
        self.obs = obs

    def __eq__(self, other):
        return np.array_equal(self.state, other.state)

    def __hash__(self):
        return hash(tuple(self.state.tolist()))

    def unpack(self):
        return (self.state, self.obs)


def plan_traj(trans_prob, start_obs, goal_obs, posterior_function):
    weights = -np.log(trans_prob + 1e-8)
    # apply cutoff to very low probabilities
    cutoff = 3
    weights[weights > cutoff] = 0
    G = nx.DiGraph(weights)
    c_start = posterior_function(start_obs)
    c_goal = posterior_function(goal_obs)
    try:
        c_traj = nx.shortest_path(G, source=c_start, target=c_goal, weight='weight')
    except:
        c_traj = []  # no trajectory found
    return c_traj


class SolverNoPruning(AStar):
    """
    Use astar algorithm. 
    Node is a tuple: (binary state representation of size c_dim, observation vector)

    transition function : map current state to a sample of next state
    posterior_function : map observation to state
    discriminator_function : map two observations to confidence score (that they are from the data)
    generator_function : map current state, next state, and current observation to a sample of next observation
    """

    def __init__(self,
                 transition_function,
                 generator_function,
                 discriminator_function=None,
                 relaxation=10.0,
                 mc_samples=100):
        self.transition = transition_function
        self.generator = generator_function
        self.mc_samples = mc_samples
        self.relaxation = relaxation  # astar relaxed heuristic
        self.n_expanded = 0

    def heuristic_cost_estimate(self, n1, n2):
        """Euclidean heuristic"""
        return self.relaxation * np.linalg.norm(n1.state - n2.state)

    def distance_between(self, n1, n2):
        """minimize euclidean distance"""
        return np.linalg.norm(n1.state - n2.state)

    def is_goal_reached(self, current, goal):
        """ returns true when we can consider that 'current' is the goal"""
        return np.array_equal(current.state, goal.state)

    def neighbors(self, node):
        """
        Sample next states from current state, and generate corresponding observations.
        Use discriminator to prune impossible observation transitions
        """
        self.n_expanded += 1
        # if self.n_expanded %1 ==0:
        #     print("\tExpanded %d nodes" % self.n_expanded)
        state, observation = node.unpack()
        observations = np.tile(observation, (self.mc_samples, 1))
        states = np.tile(state, (self.mc_samples, 1))
        next_states = self.transition(states)
        next_observations = self.generator(states, next_states, observations)
        # print(next_observations)
        unique_index = []
        counts = []
        if len(next_states) > 0:
            _, unique_index, unique_inverse, counts = np.unique(next_states, return_index=True, return_inverse=True,
                                                                return_counts=True, axis=0)
        if len(counts) > 0:
            found_neighbors = [StateObsTuple(next_states[i], next_observations[i]) for i in unique_index]
        else:
            found_neighbors = []
        return found_neighbors


class Solver(AStar):
    """
    Use astar algorithm.
    Node is a tuple: (binary state representation of size c_dim, observation vector)
    transition function : map current state to a sample of next state
    posterior_function : map observation to state
    discriminator_function : map two observations to confidence score (that they are from the data)
    generator_function : map current state, next state, and current observation to a sample of next observation
    """

    def __init__(self,
                 transition_function,
                 discriminator_function,
                 generator_function,
                 discriminator_confidence_cutoff=0.7,
                 mc_samples=100,
                 relaxation=1):
        self.transition = transition_function
        self.discriminator = discriminator_function
        self.generator = generator_function
        self.mc_samples = mc_samples
        self.discriminator_confidence_cutoff = discriminator_confidence_cutoff
        self.relaxation = relaxation  # astar relaxed heuristic
        self.n_expanded = 0
        # raise DeprecationWarning('This class is not used or actively maintained. --Ge')

    def heuristic_cost_estimate(self, n1, n2):
        """No heuristic for now"""
        return self.relaxation * np.linalg.norm(n1.state - n2.state)

    def distance_between(self, n1, n2):
        """this method always returns 1, meaning that we minimize *number* of transitions"""
        return np.linalg.norm(n1.state - n2.state)

    def is_goal_reached(self, current, goal):
        """ returns true when we can consider that 'current' is the goal"""
        return np.array_equal(current.state, goal.state)

    def neighbors(self, node):
        """
        Sample next states from current state, and generate_pairs corresponding observations.
        Use discriminator to prune impossible observation transitions
        """
        self.n_expanded += 1
        if self.n_expanded % 100 == 0:
            print("\tExpanded %d nodes" % self.n_expanded)
        state, observation = node.unpack()
        observations = np.tile(observation, (self.mc_samples, 1, 1, 1))
        states = np.tile(state, (self.mc_samples, 1))
        next_states = self.transition(states)
        next_observations = self.generator(states, next_states, observations)
        confidences = self.discriminator(observations, next_observations).reshape(-1)
        # prune low confidence transitions
        states = states[confidences > self.discriminator_confidence_cutoff]
        next_states = next_states[confidences > self.discriminator_confidence_cutoff]
        inds = confidences > self.discriminator_confidence_cutoff
        next_observations = next_observations[inds]
        confidences = confidences[inds]
        unique_index = []
        unique_inverse = []
        counts = []
        if len(next_states) > 0:
            _, unique_index, unique_inverse, counts = np.unique(next_states, return_index=True, return_inverse=True,
                                                                return_counts=True, axis=0)
            # print(counts)
        if len(counts) > 0:
            found_neighbors = [[] for i in range(len(unique_index))]
            for i in range(len(unique_index)):
                inds = np.nonzero(unique_inverse == i)
                max_ind = inds[0][np.argmax(confidences[inds])]
                assert max_ind.size == 1
                found_neighbors[i] = StateObsTuple(next_states[max_ind], next_observations[max_ind])
                # found_neighbors = [StateObsTuple(next_states[i], next_observations[i]) for i in unique_index[np.array(counts) >= 3]]
        else:
            found_neighbors = []
        return found_neighbors


def plan_traj_astar(
        start_obs,
        goal_obs,
        start_state,
        goal_state,
        preprocess_function,
        transition_function,
        discriminator_function,
        generator_function,
        solver_class=SolverNoPruning, verbose=False, **kwargs):
    # todo: this is completely not necessary. We should pass an instance of the solver not a class.
    """
    Use astar to plan a trajectory from start_obs to goal_obs.

    :param transition_function : map current state to a sample of next state
    :param posterior_function : map observation to state Q(ob | s)
    :param discriminator_function : map two observations to confidence score (that they are from the data) D(f | ob_1, ob_2)
    :param generator_function : map current state, next state, and current observation to a sample of next observation
    :param solver_class:
    :param discriminator_confidence_cutoff:
    :param **kwargs: pass the rest into the solver.
    """
    start_state = preprocess_function(start_state)
    goal_state = preprocess_function(goal_state)
    solver = solver_class(transition_function=transition_function,
                          discriminator_function=discriminator_function,
                          generator_function=generator_function,
                          **kwargs)
    s, g = StateObsTuple(start_state, start_obs), StateObsTuple(goal_state, goal_obs)
    foundPath = solver.astar(s, g)
    if verbose and foundPath:
        print("start state: ", start_state)
        print("goal state: ", goal_state)
        print("Plan length: ", len(foundPath) + 2)
        for j, obj in enumerate(([s] + list(foundPath) + [g])):
            print("\t", obj.state)
            # print(undiscretize(obj.state, self.discretization_bins, self.P.unif_range))
    elif verbose:
        print("Path not found!")
    return list(foundPath) if foundPath else None
