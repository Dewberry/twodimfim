#!/bin/bash

set -euo pipefail

source .env

aws ecr get-login-password --region ${ECR_AWS_REGION} | docker login --username AWS --password-stdin ${ECR_AWS_ACCOUNT_ID}.dkr.ecr.${ECR_AWS_REGION}.amazonaws.com

docker compose -f docker-compose.yml pull
docker compose -f docker-compose.yml up -d
docker compose -f docker-compose.yml logs -f
