"""
    author: SPDKH
    date: Nov 2, 2023
"""
from src import init


def main():
    """
        Main Predicting Function
    """
    dnn = init.run()

    dnn.predict()
    print("\n [*] Testing finished!")


if __name__ == '__main__':
    main()
