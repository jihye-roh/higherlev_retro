export GATEWAY_URL=http://0.0.0.0:9100
export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core

echo Stopping service for module: tree_search_expand_one, runtime: docker
docker stop expand_one; docker rm expand_one

echo Building image for module: tree_search_expand_one, runtime: docker, device: cpu
docker build -f Dockerfile_cpu -t ${ASKCOS_REGISTRY}/tree_search/expand_one:1.0-cpu .

echo Starting service for module: tree_search_expand_one, runtime: docker, device: cpu
sh scripts/serve_cpu_in_docker.sh
