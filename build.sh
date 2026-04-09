cd sepex
docker build -t sepex-adapter:latest .

# docker run -i \
#  -v $(pwd)/scott:/mnt \
#   sepex-adapter:latest <<< \
#  '{"inputs": {"input1": {"data": "file:///mnt/entrada.csv"}}}'


docker build -f dockerfile.sepex -t twodimfim:latest .
# docker run -i \
#  -v $(pwd)/scott:/mnt \
#   twodimfim:latest <<< \
#  '{"inputs": {"input1": {"data": "file:///mnt/entrada.csv"}}}'