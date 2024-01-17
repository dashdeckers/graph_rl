use {
    graph_rl::{
        agents::{
            Algorithm,
            DDPG_SGM,
        },
        envs::{
            Environment,
            PointEnv,
            PointEnvConfig,
            PointEnvWalls,
            PointReward,
        },
        configs::{
            DDPG_SGM_Config,
            TrainConfig,
        },
        engines::{
            setup_logging,
            run_experiment_off_policy,
            ParamRunMode,
            ParamEnv,
            ParamAlg,
            SgmGUI,
        },
    },
    candle_core::{
        Device,
        CudaDevice,
        backend::BackendDevice,
    },
    anyhow::Result,
    tracing::Level,
};


fn main() -> Result<()> {
    let experiment_name = "sgm-test";

    setup_logging(
        &experiment_name,
        Some(Level::WARN),
    )?;

    // let device = Device::Cuda(CudaDevice::new(0)?);
    let device = Device::Cpu;


    //// Create the PointEnv Environment for Training ////

    let pointenv = *PointEnv::new(
        PointEnvConfig::new(
            // Some(vec![
            //     ((0.0, 5.0), (5.0, 5.0)).into(),
            //     ((5.0, 5.0), (5.0, 4.0)).into(),
            // ]),
            10.0,
            10.0,
            PointEnvWalls::None,
            30,
            1.0,
            0.5,
            None,
            0.1,
            PointReward::Distance,
            42,
        ),
    )?;


    //// Create DDPG_SGM Algorithm ////

    let ddpg_sgm = *DDPG_SGM::from_config(
        &device,
        &DDPG_SGM_Config::pointenv(),
        pointenv.observation_space().iter().product::<usize>(),
        pointenv.action_space().iter().product::<usize>(),
    )?;


    //// Check Pretrained DDPG_SGM Performance via GUI ////

    SgmGUI::<DDPG_SGM<PointEnv>, PointEnv, _, _>::open(
        ParamEnv::AsEnvironment(pointenv.clone()),
        ParamAlg::AsAlgorithm(ddpg_sgm.clone()),
        ParamRunMode::Train(TrainConfig::new(
            200,
            30,
            500,
        )),
        device.clone(),
    );


    //// Run Pretrained DDPG_SGM Algorithm in Experiment ////

    run_experiment_off_policy::<DDPG_SGM<PointEnv>, PointEnv, _, _>(
        &experiment_name,
        100,
        ParamEnv::AsEnvironment(pointenv.clone()),
        ParamAlg::AsAlgorithm(ddpg_sgm),
        ParamRunMode::Train(TrainConfig::new(
            200,
            30,
            500,
        )),
        &device,
    )?;


    Ok(())
}