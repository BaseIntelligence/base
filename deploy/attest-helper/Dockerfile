# syntax=docker/dockerfile:1.7
# Minimal miner CVM attest-helper: GET /v1/quote → dstack GetQuote + D10 report_data.
# Pin public Python runtime by digest (no :latest).
FROM python@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b
WORKDIR /app
COPY server.py /app/server.py
USER nobody
ENV GBASE_ATTEST_PORT=8081 \
    GBASE_DSTACK_SOCK=/var/run/dstack.sock
EXPOSE 8081
ENTRYPOINT ["python3", "-u", "/app/server.py"]
