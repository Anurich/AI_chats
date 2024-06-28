import logging

class CustomLogger:
    """
    A custom logger class to encapsulate logging setup and functionality.
    """

    def __init__(self, name, level=logging.DEBUG):
        """
        Initializes the CustomLogger instance.

        Args:
            name (str): The name of the logger.
            level (int): The logging level. Default is logging.DEBUG.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create console handler and set level to INFO
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Add formatter to ch
        ch.setFormatter(formatter)

        # Add ch to logger
        self.logger.addHandler(ch)

    def log_debug(self, message):
        """
        Log a message at DEBUG level.

        Args:
            message (str): The message to log.
        """
        self.logger.debug(message)

    def log_info(self, message):
        """
        Log a message at INFO level.

        Args:
            message (str): The message to log.
        """
        self.logger.info(message)

    def log_warning(self, message):
        """
        Log a message at WARNING level.

        Args:
            message (str): The message to log.
        """
        self.logger.warning(message)

    def log_error(self, message):
        """
        Log a message at ERROR level.

        Args:
            message (str): The message to log.
        """
        self.logger.error(message)

    def log_critical(self, message):
        """
        Log a message at CRITICAL level.

        Args:
            message (str): The message to log.
        """
        self.logger.critical(message)
