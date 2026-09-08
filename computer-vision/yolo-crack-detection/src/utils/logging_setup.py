import os

from python_logger import LoggerConfig, New


_LOGGER_FACTORY = New(
    LoggerConfig(
        service_name="yolo-crack-detection",
        version="1.0.0",
        stand_domain=os.getenv("STAND_DOMAIN", "local"),
        log_file_path=os.getenv("LOG_FILE", ""),
        trace_enabled=os.getenv("LOGGER_TRACE_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317"),
        color=None,
    )
)


def get_logger(name: str):
    return _LOGGER_FACTORY.get_logger(name)


def shutdown_logging() -> None:
    _LOGGER_FACTORY.force_flush()
    _LOGGER_FACTORY.close()
