import logging 

def setup_logger(log_file):
    print(f"Log file: {log_file}")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a formatter for file logging with timestamp
    file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    # Create a formatter for console logging without timestamp
    console_formatter = logging.Formatter('%(message)s')

    # Create a console handler (prints to console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # Create a file handler (writes to a file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)