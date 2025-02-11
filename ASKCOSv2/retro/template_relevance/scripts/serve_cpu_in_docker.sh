#!/bin/bash

if [ -z "${ASKCOS_REGISTRY}" ]; then
  export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core
fi

if [ "$(docker ps -aq -f status=exited -f name=^retro_template_relevance$)" ]; then
  # cleanup if container died;
  # otherwise it would've been handled by make stop already
  docker rm retro_template_relevance
fi

docker run -d \
  --name retro_template_relevance \
  -p 9410-9412:9410-9412 \
  -v "$PWD/mars":/app/template_relevance/mars \
  -t "${ASKCOS_REGISTRY}"/retro/template_relevance:1.0-cpu \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=/app/template_relevance/mars \
  --models \
  uspto_higher-level_consol=uspto_higher-level_consol.mar \
  uspto_higher-level_nonconsol=uspto_higher-level_nonconsol.mar \
  uspto_original_consol=uspto_original_consol.mar \
  uspto_original_nonconsol=uspto_original_nonconsol.mar \
  --ts-config ./config.properties
