import logging
import os

def setup_logger(log_file='app.log'):
    logger = logging.getLogger("TransportLogger")
    logger.setLevel(logging.INFO)

    # Crée un handler fichier
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Format de log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Ajoute le handler au logger
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger
