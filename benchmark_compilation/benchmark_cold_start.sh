set -ex

docker build -t benchmark-cold-start -f benchmark_compilation/Dockerfile .

docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    benchmark-cold-start
