use aquamarine::aquamarine;
use candle_core::{
    DType,
    Device,
    Result,
    Tensor,
};

#[cfg_attr(doc, aquamarine)]
/// ```mermaid
/// graph LR
///     s([Source]) --> a[[aquamarine]]
///     r[[rustdoc]] --> f([Docs w/ Mermaid!])
///     subgraph rustc[Rust Compiler]
///     a -. inject mermaid.js .-> r
///     end
/// ```
/// The Ornstein-Uhlenbeck process, given by the equation:
/// ```math
/// dX_t = \kappa(\theta-X_t)dt + \sigma dW_t
/// ```
/// Where in our code we should replace theta with kappa
pub struct OuNoise {
    mu: f64,
    theta: f64,
    sigma: f64,
    state: Tensor,
}
impl OuNoise {
    pub fn new(
        mu: f64,
        theta: f64,
        sigma: f64,
        size_action: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            mu,
            theta,
            sigma,
            state: Tensor::ones(size_action, DType::F64, device)?,
        })
    }

    pub fn sample(&mut self) -> Result<Tensor> {
        let rand = Tensor::randn_like(&self.state, 0.0, 1.0)?;
        let dx = ((self.theta * (self.mu - &self.state)?)? + (self.sigma * rand)?)?;
        self.state = (&self.state + dx)?;
        Ok(self.state.clone())
    }
}
