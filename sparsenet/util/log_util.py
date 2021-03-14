# Created at 2020-06-04
# Summary: log related functions

# myapp.py
import logging

def main():
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )
    # logging.basicConfig(level=logging.INFO)
    logging.info('Started')
    # mylib.do_something()
    logging.info('Finished')
    logging.warning('Watch out!')  # will print a message to the console


if __name__ == '__main__':
    main()