from __future__ import annotations


def init_otel(service_name: str) -> None:
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        trace.set_tracer_provider(
            TracerProvider(resource=Resource.create({"service.name": service_name}))
        )
    except Exception:
        return
