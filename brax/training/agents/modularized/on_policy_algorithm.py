import typing
import jax
import brax

# TODO: Switch to jaxtyping instead of raw typing in order to make this stuff much more sensible

# These aren't needed I just like them :)
def tree_stack(trees):
    return jax.tree_util.tree_map(lambda *v: jax.numpy.stack(v), *trees)

def tree_unstack(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]

def print_tree_shape(tree):
    [print(key_path, array.shape) for key_path, array in jax.tree_util.tree_leaves_with_path(tree, lambda x: isinstance(x, jax.Array))]

PyTree = typing.Any
TrainingState = PyTree
PolicyState = PyTree
AgentState = PyTree
EnvState = brax.envs.State
class WorldState(typing.NamedTuple):
    agent_state: AgentState
    env_state: EnvState

# Trajectory = typing.List[typing.Tuple[WorldState, WorldState, jax.Array]]
class Transition(typing.NamedTuple):
    current_world_state: WorldState
    next_world_state: WorldState
    action: PyTree # this is a pytree that is the action emitted by the policy

Trajectory = typing.List[Transition]

class TransitionState(typing.NamedTuple):
    world_state: WorldState
    key: jax.Array

class AlgorithmState(typing.NamedTuple):
    training_state: TrainingState
    policy_state: PolicyState
    key: jax.Array

# the policy function should not need the policy state. Instead it should be functools.partial'ed over the policy state. This should be something like a make policy function generates the function
# for symmetry, this would mean that we also have make_train_fn and make_reset_fn, but those don't have static state. This feels like code smell because of the lack of symmetry but I'm too lazy
# to decide if it's real code smell or if there's something different about policy function.
PolicyFunction = typing.Callable[[PolicyState, WorldState, jax.Array], typing.Tuple[WorldState, PyTree]]
TrainingFunction = typing.Callable[[TrainingState, PolicyState, typing.List[Transition], jax.Array], typing.Tuple[TrainingState, PolicyState, PyTree]]
ResetFunction = typing.Callable[[jax.Array], WorldState]

"""
Lightweight and composable on policy algorithm implementation.

Motivation: Being able to wrap existing algorithms and extend them in interesting ways. Currently, algorithms in brax training are implemented in a single
function. This works pretty well, but makes it very difficult to extend the algorithm with different techniques. By standarizing the two phases of on-policy
algorithms, trajectory collection and policy optimization, we can implement algorithms that have clear implementations.

* Jitting *
You must jit this function. 

* Environment *
The environment must be wrapped with auto reset, otherwise, the parallelized trajectory collection would fail.

* Metrics *
We jit everything here in order to allow for maximum performance. If you don't like this, use pytorch. To do debugging, logging, and printing,
you should use the jax APIs associated with that. Specifically: jax.debug.callback, jax.debug.print, jax.debug.breakpoint

* vmapping *
You don't need to vmap your environment because we are going to vmap it with the on policy algorithm as part of the environment rollouts. 
This choice was taken to allow for pmapping inside this algorithm framework
"""
def on_policy_algorithm(
    # injected dependencies
    policy_fn: PolicyFunction,
    train_fn: TrainingFunction,
    reset_fn: ResetFunction,

    # training parameters
    num_transitions: int, # timesteps per episode
    num_trajectories: int, # total number of trajectories per epoch
    num_updates: int, # number of training steps to take per epoch

    # state
    algorithm_state: AlgorithmState,
) -> AlgorithmState:
    @jax.jit
    def transition_fn(
        transition_state: TransitionState,
        _
    ) -> typing.Tuple[TransitionState, Transition]:
        next_world_state, action = policy_fn(algorithm_state.policy_state, transition_state.world_state, transition_state.key)
        next_key = jax.random.split(transition_state.key, 1)[0]
        return TransitionState(
            world_state=next_world_state,
            key=next_key
        ), Transition(
            current_world_state=transition_state.world_state,
            next_world_state=next_world_state,
            action=action
        )
    
    @jax.jit
    def trajectory_fn(
        key: jax.Array,
    ):
        reset_key, transition_key = jax.random.split(key)
        _, trajectory = jax.lax.scan(
            transition_fn,
            TransitionState(
                world_state=reset_fn(reset_key),
                key=transition_key
            ),
            (),
            length=num_transitions,
        )
        return trajectory
    
    def update_fn(
        carry: typing.Tuple[TrainingState, PolicyState, jax.Array],
        _
    ) -> typing.Tuple[TrainingState, PolicyState, jax.Array]:
        current_training_state, current_policy_state, current_key = carry
        (next_training_state, next_policy_state) = train_fn(
            current_training_state,
            current_policy_state,
            trajectories,
        )
        next_key, _ = jax.random.split(current_key, 2)
        return (next_training_state, next_policy_state, next_key), None
    
    next_key, trajectory_key, training_key = jax.random.split(algorithm_state.key, 3)
    trajectories = jax.vmap(trajectory_fn)(
        jax.random.split(trajectory_key, num_trajectories) # this will parallelize the trajectory collection
    )

    (next_training_state, next_policy_state, _), _ = jax.lax.scan(
        update_fn,
        (algorithm_state.training_state, algorithm_state.policy_state, training_key),
        (),
        length=num_updates
    )

    return AlgorithmState(
        training_state=next_training_state,
        policy_state=next_policy_state,
        key=next_key
    )