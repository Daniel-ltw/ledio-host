import argparse

def main():
    parser = argparse.ArgumentParser(description="This is a command line tool for doing something useful with a llm.")
    subparsers = parser.add_subparsers(dest='command', help="Sub-command to execute.")
    subparsers.add_parser('something', help="This is an example of how to add arguments to your tool")
    args = parser.parse_args()

    if args.command == 'something':
        print('something indeed')
    else:
        print('It looks like you need help')
        parser.print_help()
