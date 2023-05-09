import argparse
from richgan.utils.tbreader import TBAggregator

parser = argparse.ArgumentParser()
parser.add_argument("--pattern", type=str, required=True)
parser.add_argument("-o", type=str, required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    aggregator = TBAggregator(args.pattern)
    result = aggregator.aggregate()
    result.to_csv(args.o, index=False)
