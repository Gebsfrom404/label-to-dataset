import sys
import warnings
import logging

from ltd.app import create_application
from ltd.main_window import MainWindow


def suppress_warnings():
    environment = __import__('os').getenv('LTD_ENVIRONMENT')
    if environment == 'development':
        return
    logging.basicConfig(level=logging.ERROR)
    warnings.simplefilter('ignore')


def main():
    suppress_warnings()
    app = create_application()
    window = MainWindow(app)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
