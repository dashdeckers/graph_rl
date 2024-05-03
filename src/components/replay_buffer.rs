use {
    crate::envs::TensorConvertible,
    candle_core::{
        Result,
        Tensor,
    },
    rand::{
        distributions::Uniform,
        thread_rng,
        Rng,
    },
    std::collections::VecDeque,
    unzip_n::unzip_n,
};

unzip_n!(6);

/// A transition in the replay buffer.
///
/// # Fields
///
/// * `state` - The state tensor.
/// * `action` - The action tensor.
/// * `reward` - The reward tensor.
/// * `next_state` - The next state tensor.
/// * `terminated` - The terminated tensor.
/// * `truncated` - The truncated tensor.
#[derive(Clone)]
pub struct Transition {
    state: Tensor,
    action: Tensor,
    reward: Tensor,
    next_state: Tensor,
    terminated: Tensor,
    truncated: Tensor,
}
impl Transition {
    fn new(
        state: &Tensor,
        action: &Tensor,
        reward: &Tensor,
        next_state: &Tensor,
        terminated: &Tensor,
        truncated: &Tensor,
    ) -> Self {
        Self {
            state: state.clone(),
            action: action.clone(),
            reward: reward.clone(),
            next_state: next_state.clone(),
            terminated: terminated.clone(),
            truncated: truncated.clone(),
        }
    }
}

/// A replay buffer for off-policy algorithms.
///
/// The replay buffer is implemented as a simple ring buffer / VecDeque.
///
/// # Fields
///
/// * `buffer` - The buffer of transitions.
/// * `capacity` - The capacity of the buffer.
/// * `size` - The current size of the buffer.
#[derive(Clone)]
pub struct ReplayBuffer {
    buffer: VecDeque<Transition>,
    capacity: usize,
    size: usize,
}
impl ReplayBuffer {
    /// Create a new replay buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            size: 0,
        }
    }

    /// Check if the buffer is full.
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    /// Set capacity
    pub fn set_capacity(
        &mut self,
        capacity: usize,
    ) {
        self.capacity = capacity;
    }

    /// Push a transition into the buffer.
    ///
    /// If the buffer is full, the oldest transition is removed to make room for
    /// the new transition.
    pub fn push(
        &mut self,
        state: &Tensor,
        action: &Tensor,
        reward: &Tensor,
        next_state: &Tensor,
        terminated: &Tensor,
        truncated: &Tensor,
    ) {
        if self.size == self.capacity {
            self.buffer.pop_front();
        } else {
            self.size += 1;
        }
        self.buffer.push_back(Transition::new(
            state, action, reward, next_state, terminated, truncated,
        ));
    }

    /// Sample a random batch of transitions from the buffer.
    ///
    /// When the size of the buffer is less than the batch size, `None` is returned.
    #[allow(clippy::type_complexity)]
    pub fn random_batch(
        &self,
        batch_size: usize,
    ) -> Result<Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)>> {
        if self.size < batch_size {
            Ok(None)
        } else {
            let transition_to_tuple =
                |t: &Transition| -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
                    Ok((
                        t.state.unsqueeze(0)?,
                        t.action.unsqueeze(0)?,
                        t.reward.unsqueeze(0)?,
                        t.next_state.unsqueeze(0)?,
                        t.terminated.unsqueeze(0)?,
                        t.truncated.unsqueeze(0)?,
                    ))
                };

            let transitions: Vec<&Transition> = thread_rng()
                .sample_iter(Uniform::from(0..self.size))
                .take(batch_size)
                .map(|i| self.buffer.get(i).unwrap())
                .collect();

            let (states, actions, rewards, next_states, terminateds, truncateds) =
                transitions
                .into_iter()
                .map(transition_to_tuple)
                .collect::<Result<Vec<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)>>>()?
                .into_iter()
                .unzip_n_vec();

            Ok(Some((
                Tensor::cat(&states, 0)?,
                Tensor::cat(&actions, 0)?,
                Tensor::cat(&rewards, 0)?,
                Tensor::cat(&next_states, 0)?,
                Tensor::cat(&terminateds, 0)?,
                Tensor::cat(&truncateds, 0)?,
            )))
        }
    }

    /// Get all states in the buffer as `Observation`s.
    ///
    /// This collects all the [`Tensor`] states in the buffer and returns them
    /// as `Observation`s.
    pub fn all_states<S: TensorConvertible>(&self) -> Vec<S> {
        let mut states: Vec<S> = self
            .buffer
            .iter()
            .map(|t| <S>::from_tensor(t.state.clone()))
            .collect();

        states.extend(
            self.buffer
                .back()
                .map(|t| <S>::from_tensor(t.next_state.clone())),
        );

        states
    }
}
