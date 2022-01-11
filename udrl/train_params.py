class TrainParams:

    def __init__(
            self,
            n_main_iter=1000,
            batch_size=768,
            horizon_scale=0.01,
            return_scale=0.01,
            last_few=75,
            learning_rate=0.0008,
            n_episodes_per_iter=20,
            n_updates_per_iter=500,
            n_warm_up_episodes=30,
            skip_every_n_observations=3,
            replay_size=400,
            evaluate_every=5,
            target_return=500,
            min_reward=-15,
            max_reward=15,
            max_steps=1000,
            max_steps_penalty=-50,
            n_evals=1,
            stop_on_solved=False,
            save_on_eval=False
    ):
        self.n_main_iter = n_main_iter
        self.batch_size = batch_size
        self.horizon_scale = horizon_scale
        self.return_scale = return_scale
        self.last_few = last_few
        self.learning_rate = learning_rate
        self.n_episodes_per_iter = n_episodes_per_iter
        self.n_updates_per_iter = n_updates_per_iter
        self.n_warm_up_episodes = n_warm_up_episodes
        self.skip_every_n_observations = skip_every_n_observations
        self.replay_size = replay_size
        self.evaluate_every = evaluate_every
        self.target_return = target_return
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.max_steps = max_steps
        self.max_steps_penalty = max_steps_penalty
        self.n_evals = n_evals
        self.stop_on_solved = stop_on_solved
        self.save_on_eval = save_on_eval
