use std::collections::VecDeque;

use rand::{distributions::Uniform, thread_rng, Rng};
use candle_core::{Tensor, Result};

use crate::envs::TensorConvertible;


#[derive(Clone)]
struct Transition {
    state: Tensor,
    action: Tensor,
    reward: Tensor,
    next_state: Tensor,
    terminated: bool,
    truncated: bool,
}
impl Transition {
    fn new(
        state: &Tensor,
        action: &Tensor,
        reward: &Tensor,
        next_state: &Tensor,
        terminated: bool,
        truncated: bool,
    ) -> Self {
        Self {
            state: state.clone(),
            action: action.clone(),
            reward: reward.clone(),
            next_state: next_state.clone(),
            terminated,
            truncated,
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
        terminated: bool,
        truncated: bool,
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
    ) -> Result<Option<(Tensor, Tensor, Tensor, Tensor, Vec<bool>, Vec<bool>)>> {
        if self.size < batch_size {
            Ok(None)
        } else {
            let transitions: Vec<&Transition> = thread_rng()
                .sample_iter(Uniform::from(0..self.size))
                .take(batch_size)
                .map(|i| self.buffer.get(i).unwrap())
                .collect();

            let states: Vec<Tensor> = transitions
                .iter()
                .map(|t| t.state.unsqueeze(0))
                .collect::<Result<_>>()?;
            let actions: Vec<Tensor> = transitions
                .iter()
                .map(|t| t.action.unsqueeze(0))
                .collect::<Result<_>>()?;
            let rewards: Vec<Tensor> = transitions
                .iter()
                .map(|t| t.reward.unsqueeze(0))
                .collect::<Result<_>>()?;
            let next_states: Vec<Tensor> = transitions
                .iter()
                .map(|t| t.next_state.unsqueeze(0))
                .collect::<Result<_>>()?;
            let terminateds: Vec<bool> = transitions.iter().map(|t| t.terminated).collect();
            let truncateds: Vec<bool> = transitions.iter().map(|t| t.truncated).collect();

            Ok(Some((
                Tensor::cat(&states, 0)?,
                Tensor::cat(&actions, 0)?,
                Tensor::cat(&rewards, 0)?,
                Tensor::cat(&next_states, 0)?,
                terminateds,
                truncateds,
            )))
        }
    }

    pub fn all_states<S: TensorConvertible>(&self) -> Vec<S> {
        let mut states: Vec<S> = self.buffer
            .iter()
            .map(|t| <S>::from_tensor(t.state.clone()))
            .collect();

        states.extend(self.buffer
            .back()
            .map(|t| <S>::from_tensor(t.next_state.clone()))
        );

        states
    }
}
