"""
    author: SPDKH
    date: Nov 2, 2023
"""
from src import init


def main():
    """
        Main Training Function
    """
    # build graph
    dnn = init.run()

    # show network architecture
    # show_all_variables()
    #
    # launch the graph in a session
    dnn.train()
    print("\n [*] Training finished!")

    dnn.predict()
    print("\n [*] Testing finished!")


if __name__ == '__main__':
    main()
