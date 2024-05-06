$env:RUST_BACKTRACE=1;

$REPS = 10;


cargo run `
    --release `
    --example pointenv_ddpg `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_empty_mindist.ron" `
    --alg-config ".\examples\configs\ddpg.ron" `
    --n-repetitions $REPS `
    --log warn `
    --name ddpg-pointenv-empty-mindist `
    &


cargo run `
    --release `
    --example pointenv_ddpg `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_empty_maxdist.ron" `
    --alg-config ".\examples\configs\ddpg.ron" `
    --n-repetitions $REPS `
    --log warn `
    --name ddpg-pointenv-empty-maxdist `
    &

