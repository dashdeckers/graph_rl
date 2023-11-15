use candle_core::{
    DType,
    Device,
    Result,
    Tensor,
};

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
