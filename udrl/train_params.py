class TrainParams:

    def __init__(
            self,
            n_main_iter=100,
            batch_size=768,
            horizon_scale=0.01,
            last_few=75,
            learning_rate=0.0003,
            n_episodes_per_iter=20,
            n_updates_per_iter=100,
            n_warm_up_episodes=10,
            replay_size=500,
            return_scale=0.02,
            evaluate_every=10,
            target_return=200,
            max_reward=250,
            max_steps=300,
            max_steps_reward=-50,
            hidden_size=32,
            n_evals=1,
            stop_on_solved=False
    ):
        self.n_main_iter = n_main_iter
        self.batch_size = batch_size
        self.horizon_scale = horizon_scale
        self.last_few = last_few
        self.learning_rate = learning_rate
        self.n_episodes_per_iter = n_episodes_per_iter
        self.n_updates_per_iter = n_updates_per_iter
        self.n_warm_up_episodes = n_warm_up_episodes
        self.replay_size = replay_size
        self.return_scale = return_scale
        self.evaluate_every = evaluate_every
        self.target_return = target_return
        self.max_reward = max_reward
        self.max_steps = max_steps
        self.max_steps_reward = max_steps_reward
        self.hidden_size = hidden_size
        self.n_evals = n_evals
        self.stop_on_solved = stop_on_solved
