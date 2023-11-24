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

#[derive(Clone)]
struct Transition {
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

pub struct ReplayBuffer {
    buffer: VecDeque<Transition>,
    capacity: usize,
    size: usize,
}
impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            size: 0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

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
