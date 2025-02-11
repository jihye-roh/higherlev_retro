#!/bin/bash

if [ -z "${ASKCOS_REGISTRY}" ]; then
  export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core
fi

if [ -z "${GATEWAY_URL}" ]; then
  # empty GATEWAY_URL means it's not passed in from core .env;
  # probably development mode
  export GATEWAY_URL=http://0.0.0.0:9100
fi

if [ "$(docker ps -aq -f status=exited -f name=^mcts$)" ]; then
  # cleanup if container died;
  # otherwise it would've been handled by make stop already
  docker rm mcts
fi

docker run -d \
  --name mcts \
  --env GATEWAY_URL="$GATEWAY_URL" \
  --network=host \
  -t ${ASKCOS_REGISTRY}/tree_search/mcts:1.0-cpu
