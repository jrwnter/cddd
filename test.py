import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cell_size', nargs='+', default=[128], type=int)
    flags, unparsed = parser.parse_known_args()
    print(flags.cell_size)