#[derive(Clone)]
pub struct TrainingConfig {
    // The learning rates for the Actor and Critic networks
    pub actor_learning_rate: f64,
    pub critic_learning_rate: f64,
    // The impact of the q value of the next state on the current state's q value.
    pub gamma: f64,
    // The weight for updating the target networks.
    pub tau: f64,
    // The number of neurons in the hidden layers of the Actor and Critic networks.
    pub hidden_1_size: usize,
    pub hidden_2_size: usize,
    // The capacity of the replay buffer used for sampling training data.
    pub replay_buffer_capacity: usize,
    // The training batch size for each training iteration.
    pub training_batch_size: usize,
    // The total number of episodes.
    pub max_episodes: usize,
    // The maximum length of an episode.
    pub episode_length: usize,
    // The number of training iterations after one episode finishes.
    pub training_iterations: usize,
    // Number of random actions to take at very beginning of training.
    pub initial_random_actions: usize,
    // Ornstein-Uhlenbeck process parameters.
    pub ou_mu: f64,
    pub ou_theta: f64,
    pub ou_sigma: f64,
    // Sparse Graphical Memory parameters
    pub sgm_freq: usize,
    pub sgm_maxdist: f64,
    pub sgm_tau: f64,
}
impl TrainingConfig {
    pub fn pendulum() -> Self {
        Self {
            actor_learning_rate: 1e-4,
            critic_learning_rate: 1e-3,
            gamma: 0.99,
            tau: 0.005,
            hidden_1_size: 400,
            hidden_2_size: 300,
            replay_buffer_capacity: 100_000,
            training_batch_size: 100,
            max_episodes: 30,
            episode_length: 200,
            training_iterations: 200,
            initial_random_actions: 0,
            ou_mu: 0.0,
            ou_theta: 0.15,
            ou_sigma: 0.1,
            sgm_freq: 0,
            sgm_maxdist: 1.0,
            sgm_tau: 0.4,
        }
    }

    pub fn pointenv(timelimit: usize) -> Self {
        Self {
            actor_learning_rate: 0.0003,     //1e-4,
            critic_learning_rate: 0.0003,    //1e-3,
            gamma: 1.0,                      //0.99,
            tau: 0.005,                      //0.005,
            hidden_1_size: 256,              //400,
            hidden_2_size: 256,              //300,
            replay_buffer_capacity: 100_000, //100,
            training_batch_size: 64,         //100,
            max_episodes: 30,
            episode_length: timelimit,
            training_iterations: 30,      //200,
            initial_random_actions: 1000, //100,
            ou_mu: 0.0,
            ou_theta: 0.15, //2.0, //0.15,
            ou_sigma: 0.1,  //0.8, //0.1,
            sgm_freq: 0,
            sgm_maxdist: 1.0,
            sgm_tau: 0.4,
        }
    }
}
