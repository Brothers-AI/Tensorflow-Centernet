import argparse

def parse_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--config', required=True,
                        type=str, help='Config file path')
    return parser.parse_args()
