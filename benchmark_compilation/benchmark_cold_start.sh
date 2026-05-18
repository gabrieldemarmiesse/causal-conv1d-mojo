set -ex

docker build -t benchmark-cold-start -f benchmark_compilation/Dockerfile .

docker run --rm --gpus all benchmark-cold-start
