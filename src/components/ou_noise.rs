use candle_core::{
    DType,
    Device,
    Result,
    Tensor,
};

/// The Ornstein-Uhlenbeck process.
///
/// This process generates a noise that is correlated with the previous noise,
/// and is given by the following equation:
///
/// ```math
/// dX_t = \kappa(\theta-X_t)dt + \sigma dW_t
/// ```
///
/// An simple explanation of the parameters is given in
/// [this](https://quant.stackexchange.com/q/17590)
/// StackExchange question, by
/// [this](https://quant.stackexchange.com/a/3047)
/// answer:
///
/// ---
/// `$\theta$` is the "mean" for this process. If `$X_t > \theta \implies (\theta - X_t) < 0 $`,
/// which means that the drift for the process is negative and tends towards `$\theta$`.
/// The opposite case can be made for `$X_t < \theta$` ; the process will have positive drift
/// when `$X_t$` is below `$\theta$`.
///
/// Therefore we can consider `$\kappa$` to be the "speed" of mean reversion, scaling the
/// distance between `$X_t$` and `$\theta$` appropriately to match whatever is being modeled.
///
/// `$\sigma dW_t $` is your standard Wiener process scaled by volatility `$\sigma$`.
///
/// In plain English you can interpret the differentials as a process that reverts to a mean,
/// `$\theta$`, with speed, `$\kappa$`, and volatility, `$\sigma$`.
///
/// ---
pub struct OuNoise {
    mu: f64,
    kappa: f64,
    sigma: f64,
    state: Tensor,
}
impl OuNoise {
    pub fn new(
        mu: f64,
        kappa: f64,
        sigma: f64,
        size_action: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            mu,
            kappa,
            sigma,
            state: Tensor::ones(size_action, DType::F64, device)?,
        })
    }

    pub fn sample(&mut self) -> Result<Tensor> {
        let rand = Tensor::randn_like(&self.state, 0.0, 1.0)?;
        let dx = ((self.kappa * (self.mu - &self.state)?)? + (self.sigma * rand)?)?;
        self.state = (&self.state + dx)?;
        Ok(self.state.clone())
    }
}
