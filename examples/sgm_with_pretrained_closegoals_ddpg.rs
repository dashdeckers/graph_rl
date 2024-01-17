use {
    graph_rl::{
        agents::{
            Algorithm,
            DDPG,
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
            DDPG_Config,
            DDPG_SGM_Config,
            TrainConfig,
        },
        engines::{
            setup_logging,
            run_experiment_off_policy,
            loop_off_policy,
            ParamRunMode,
            ParamEnv,
            ParamAlg,
            SgmGUI,
        },
        components::sgm::DistanceMode,
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
    let experiment_name = "sgm-with-pretrained-closegoals-ddpg";

    setup_logging(
        &experiment_name,
        Some(Level::WARN),
    )?;

    let device = Device::Cuda(CudaDevice::new(0)?);
    // let device = Device::Cpu;


    //// Create the PointEnv Environment for Pretraining ////

    let mut pretrain_env = *PointEnv::new(
    PointEnvConfig::new(
            // Some(vec![
            //     ((0.0, 5.0), (5.0, 5.0)).into(),
            //     ((5.0, 5.0), (5.0, 4.0)).into(),
            // ]),
            5.0,
            5.0,
            PointEnvWalls::None,
            30,
            1.0,
            0.5,
            Some(2.5),
            0.1,
            PointReward::Distance,
            42,
        )
    )?;


    //// Create DDPG Algorithm for Pretraining ////

    let ddpg_config = DDPG_Config::new(
        0.0003,
        0.0003,
        1.0,
        0.005,
        256,
        256,
        1_000,
        64,
        0.0,
        0.15,
        0.2,
    );
    let mut ddpg = *DDPG::from_config(
        &device,
        &ddpg_config,
        pretrain_env.observation_space().iter().product::<usize>(),
        pretrain_env.action_space().iter().product::<usize>(),
    )?;


    //// Pretrain DDPG Algorithm ////

    let (_, successes) = loop_off_policy(
        &mut pretrain_env,
        &mut ddpg,
        ParamRunMode::Train(TrainConfig::new(
            200,
            30,
            500,
        )),
        &device,
    )?;
    println!("returns: {:#?}", successes);


    //// Create DDPG_SGM Algorithm from Pretrained DDPG for Training ////

    let ddpg_sgm_config = DDPG_SGM_Config::new(
        ddpg_config,
        DistanceMode::True,
        0.5,
        1.0,
        0.4,
    );
    let mut ddpg_sgm = *DDPG_SGM::from_config_with_ddpg(
        &device,
        &ddpg_sgm_config,
        ddpg,
    )?;
    ddpg_sgm.new_buffer(10 * 10 * 10 * 10); // 10x10 grid



    //// Define PointEnv Environment for Training ////

    let env_config = PointEnvConfig::new(
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
    );


    //// Check Pretrained DDPG_SGM Performance via GUI ////

    SgmGUI::<DDPG_SGM<PointEnv>, PointEnv, _, _>::open(
        ParamEnv::AsConfig(env_config.clone()),
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
        ParamEnv::AsConfig(env_config),
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