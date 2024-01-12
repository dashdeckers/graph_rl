size='small'
name='tensor-true'

nohup cargo run --release -- --env pointenv --alg ddpg --log warn --runs 100 --name "ddpg-$size-$name" &
nohup cargo run --release -- --env pointenv --alg ddpg-sgm --log warn --runs 100 --name "ddpg-sgm-$size-$name" &
