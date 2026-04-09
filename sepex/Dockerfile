FROM golang:1.23 AS builder

WORKDIR /build

COPY go.mod go.sum* ./
RUN go mod download 2>/dev/null || true

COPY . .

ARG TARGETARCH
RUN CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} go build -ldflags="-s -w" -o payload-writer .

FROM alpine:latest
COPY --from=builder /build/payload-writer .
ENTRYPOINT ["./payload-writer"]
