import argparse
import sys
from ledio_host.commands import host

def main():
    parser = argparse.ArgumentParser(description="Ledio Host - your radio host")
    subparsers = parser.add_subparsers(dest='command', help="Sub-command to execute.")

    # Probably could allow user to select the music host
    if len(sys.argv) == 1:
        host()
    else:
        print('Unknown command. Use -h for help.')
        parser.print_help()
