#!/usr/bin/env bash
# Generate a self-signed CA plus broker/service certificates for optional MQTT TLS (kept out of git).
set -euo pipefail
OUT=${1:-certs}
mkdir -p "${OUT:?}"
openssl req -x509 -newkey rsa:2048 -nodes -keyout "${OUT:?}/ca.key" -out "${OUT:?}/ca.crt" -days 365 -subj "/CN=drs-dev-ca" 2>/dev/null
for name in broker service; do
  openssl req -newkey rsa:2048 -nodes -keyout "${OUT:?}/${name:?}.key" -out "${OUT:?}/${name:?}.csr" -subj "/CN=${name}" 2>/dev/null
  openssl x509 -req -in "${OUT:?}/${name:?}.csr" -CA "${OUT:?}/ca.crt" -CAkey "${OUT:?}/ca.key" -CAcreateserial -out "${OUT:?}/${name:?}.crt" -days 365 2>/dev/null
  rm -f -- "${OUT:?}/${name:?}.csr"
done
echo "wrote CA and certificates to ${OUT}/ (listed in .gitignore)"
