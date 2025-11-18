#!/bin/bash

set -euo pipefail

source .env

aws ecr get-login-password --region ${ECR_AWS_REGION} | docker login --username AWS --password-stdin ${ECR_AWS_ACCOUNT_ID}.dkr.ecr.${ECR_AWS_REGION}.amazonaws.com

# docker network create process_api_net
# docker compose -f docker-compose.deploy-prod.yml down 

docker compose -f docker-compose.deploy-prod.yml pull
docker compose -f docker-compose.deploy-prod.yml up -d
docker compose -f docker-compose.deploy-prod.yml logs -f