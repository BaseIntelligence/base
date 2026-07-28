# Staging Key-Release PKI (local only)

Generate a throwaway CA + server cert for local KR. **Never commit** `config/kr/*.key` or `golden.key`.

Example generation (dev-only):

```bash
mkdir -p scripts/staging/config/kr
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout scripts/staging/config/kr/ca.key \
  -out scripts/staging/config/kr/ca.crt \
  -days 365 -subj "/CN=ac-staging-kr-ca"
# ... issue server cert for your KR host; copy ca.crt to config/kr-server-ca.crt
openssl rand -out scripts/staging/config/kr/golden.key 32
```

Real secrets stay gitignored under `config/kr/`.
