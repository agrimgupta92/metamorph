#!/bin/bash
# Launch an experiment using the docker gpu image
# Inside the metamorph folder run the following cmd:
# Usage: . scripts/run_docker_gpu.sh python metamorph/<file>.py

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line

USER_ID=`id -u`
MOUNT_DIR=''

docker run --runtime=nvidia --rm --network host --ipc=host \
    -v ${MOUNT_DIR}:/user/metamorph/output \
    -u user:${USER_ID} \
    metamorph \
    bash -c "cd /user/metamorph/ && $cmd_line"