import typing
import functools
import jax
import brax
import optax
import brax.training.agents.modularized.on_policy_algorithm as algorithm

def print_tree_shape(tree):
    [print(key_path.key, array.shape) for key_path, array in jax.tree_util.tree_leaves_with_path(tree, lambda x: isinstance(x, jax.Array))]

def print_tree(tree):
    [print(key_path, array) for key_path, array in jax.tree_util.tree_leaves_with_path(tree, lambda x: isinstance(x, jax.Array))]

class PPOAgentState(typing.NamedTuple):
    pass

class PPOPolicyState(typing.NamedTuple):
    policy_fn_params: brax.training.types.Params
    key: jax.Array

class PPOTrainingState(typing.NamedTuple):
    optimizer_params: optax.OptState
    value_fn_params: brax.training.types.Params
    key: jax.Array

class PPOAction(typing.NamedTuple):
    logits: jax.Array  # these are the logits emitted
    raw: jax.Array    # this is unprocessed and may be the same as action
    action: jax.Array  # this is the final action submitted to the environment

# create the training step for the algorithm


def make_policy(
    env: brax.envs.Env,
    key: jax.Array,

    # training params
    num_batches: int,  # number of batches to be used per training iteration

    # loss function params
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,

    # optimizer params
    learning_rate: float = 1e-4,
) -> typing.Tuple[algorithm.ResetFunction, algorithm.PolicyFunction, algorithm.TrainingFunction]:
    key, env_key = jax.random.split(key, 2)
    env_state = env.reset(env_key)
    # policy/value fn params
    ppo_network = brax.training.agents.ppo.networks.make_ppo_networks(
        env_state.obs.shape[-1],
        env.action_size,
        # do nothing with this. Second argument is normalization params
        preprocess_observations_fn=lambda observations, _: observations
    )
    make_policy = brax.training.agents.ppo.networks.make_inference_fn(ppo_network)

    # optimization
    loss_fn = functools.partial(
        brax.training.agents.ppo.losses.compute_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage
    )

    optimizer = optax.adam(learning_rate=learning_rate)

    gradient_update_fn = brax.training.gradients.gradient_update_fn(
        loss_fn,
        optimizer,
        pmap_axis_name=None,
        has_aux=True
    )

    def reset_fn(
        key: jax.Array
    ) -> algorithm.WorldState:
        # reset the environment
        env_state = env.reset(key)
        agent_state = None  # we have no explicit state that the agent carries around with it
        return algorithm.WorldState(
            agent_state=agent_state,
            env_state=env_state
        )

    def policy_fn(
        policy_state: algorithm.PolicyState,
        world_state: algorithm.WorldState,
        key: jax.Array # we don't need the key here
    ) -> typing.Tuple[algorithm.WorldState, PPOAction]:
        # take an action using the key that we got for actions
        # we don't need the policy key here because we aren't
        # doing anything stochastic
        policy_params, _ = policy_state
        policy = make_policy((None, policy_params))
        action, extras = policy(world_state.env_state.obs, key)
        next_env_state = env.step(world_state.env_state, action)
        next_actor_state = None  # this isn't using recurrent behavior so this isn't needed
        logits = extras['log_prob']
        raw = extras['raw_action']
        return algorithm.WorldState(
            agent_state=next_actor_state,
            env_state=next_env_state,
        ), PPOAction(
            logits=logits,
            raw=raw,
            action=action
        )

    def train_fn(
        training_state: algorithm.TrainingState,
        policy_state: algorithm.PolicyState,
        trajectories: typing.List[algorithm.Trajectory],
    ) -> typing.Tuple[algorithm.TrainingState, algorithm.PolicyState]:
        policy_params, policy_key = policy_state
        next_policy_key, _ = jax.random.split(policy_key, 2)
        optimizer_state, value_params, training_key, = training_state
        next_training_key, loss_key, shuffle_key = jax.random.split(training_key, 3)

        def shuffle(x: jax.Array):
            return jax.random.permutation(shuffle_key, x)
        
        def batch(x):
            return jax.numpy.reshape(x, (num_batches, x.shape[0]//num_batches, *(x.shape[1:])))
        
        # TODO: Reform losses itself to eliminate dependence on the brax transition code
        def brax_transition(
            transition: algorithm.Transition,
        ) -> brax.training.types.Transition:
            return brax.training.types.Transition(
                observation=transition.current_world_state.env_state.obs,
                action=transition.action.action,
                reward=transition.current_world_state.env_state.reward,
                discount=1-transition.next_world_state.env_state.done,
                next_observation=transition.next_world_state.env_state.obs,
                extras={
                    'state_extras': {
                        'truncation': transition.current_world_state.env_state.info['truncation'],
                    },                    
                    'policy_extras': {
                        'raw_action': transition.action.raw,
                        'log_prob': transition.action.logits
                    }
                }
            )

        def update(
            carry: typing.Tuple[optax.OptState, brax.training.types.Params, brax.training.types.Params],
            batch: typing.List[brax.training.types.Transition],
        ) -> typing.Tuple[typing.Tuple[optax.OptState, brax.training.types.Params, brax.training.types.Params], typing.Any]:
            optimizer_state, policy_params, value_params = carry
            (_, metrics), next_network_params, next_optimizer_state = gradient_update_fn(
                brax.training.agents.ppo.losses.PPONetworkParams(
                    policy=policy_params,
                    value=value_params
                ),
                None,  # There's no normalizer so this is not needed here
                batch,
                loss_key,
                optimizer_state=optimizer_state
            )
            return (next_optimizer_state, next_network_params.policy, next_network_params.value), metrics
        
        episode_mean_reward = jax.numpy.mean(trajectories.current_world_state.env_state.reward)
        episode_mean_length = jax.numpy.mean(trajectories.current_world_state.env_state.done.shape[-1] / jax.numpy.sum(trajectories.current_world_state.env_state.done, axis=-1))
        
        jax.debug.print("======== Episode Stats ========")
        jax.debug.print("Episode mean length {episode_mean_length}", episode_mean_length=episode_mean_length)
        jax.debug.print("Episode mean reward {episode_mean_reward}", episode_mean_reward=episode_mean_reward)
        
        shuffled_trajectories = jax.tree_map(shuffle, trajectories)
        batched_shuffled_trajectories = jax.tree_map(batch, shuffled_trajectories)
        brax_transitions = brax_transition(batched_shuffled_trajectories)

        (next_optimizer_state, next_policy_params, next_value_params), metrics = jax.lax.scan( # omitted return is metrics_list
            update, (optimizer_state, policy_params, value_params), brax_transitions)
        next_training_state = PPOTrainingState(
            optimizer_params=next_optimizer_state,
            value_fn_params=next_value_params,
            key=next_training_key,
        )
        next_policy_state = PPOPolicyState(
            policy_fn_params=next_policy_params,
            key=next_policy_key
        )

        jax.debug.print("======== Loss metrics ========")
        metrics_mean = jax.tree_map(lambda field: jax.numpy.mean(field), metrics)
        jax.debug.callback(print_tree, metrics_mean)

        return (next_training_state, next_policy_state)
    
    algorithm_key, policy_init_key, value_init_key, train_apply_key, policy_apply_key = jax.random.split(key, 5)
    initial_policy_params = ppo_network.policy_network.init(policy_init_key)
    initial_value_params = ppo_network.value_network.init(value_init_key)

    initial_params = brax.training.agents.ppo.losses.PPONetworkParams(
        policy=initial_policy_params,
        value=initial_value_params,
    )
    initial_algorithm_state = algorithm.AlgorithmState(
        training_state=PPOTrainingState(
            optimizer_params=optimizer.init(initial_params),
            value_fn_params=initial_value_params,
            key=train_apply_key
        ),
        policy_state=PPOPolicyState(
            policy_fn_params=initial_policy_params,
            key=policy_apply_key,
        ),
        key=algorithm_key
    )
    return reset_fn, policy_fn, train_fn, initial_algorithm_state

import tqdm
if __name__ == "__main__":
    key = jax.random.PRNGKey(seed=0)
    env = brax.envs.ant.Ant()

    key, env_key, policy_key = jax.random.split(key, 3)
    env = brax.envs.ant.Ant()
    env = brax.envs.training.EpisodeWrapper(env, 1000, 1)
    env = brax.envs.training.AutoResetWrapper(env)
    reset_fn, policy_fn, train_fn, initial_algorithm_state = make_policy(env, policy_key, num_batches=8)

    algorithm_fn = jax.jit(functools.partial(
        algorithm.on_policy_algorithm,
        policy_fn=policy_fn,
        train_fn=train_fn,
        reset_fn=reset_fn,
        num_trajectories=64,
        num_transitions=1000,
        num_updates=1,
    ))
    algorithm_state = initial_algorithm_state
    for epoch in tqdm.tqdm(range(100)):
        algorithm_state = algorithm_fn(
            algorithm_state=algorithm_state
        )