import logging

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger(name)

def log_event(logger: logging.Logger, event: str, **kwargs) -> None:
    logger.info(f'Event: {event}', extra=kwargs)

def log_error(logger: logging.Logger, error: str, error_message: str, **kwargs) -> None:
    logger.error(f'{error}: {error_message}', extra=kwargs)