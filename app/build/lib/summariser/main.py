import argparse
import sys

from dotenv import load_dotenv
load_dotenv()

from summariser.text import summarise_file
from summariser.web import summarise_web
from summariser.pdf import summarise_pdf

def main():
    parser = argparse.ArgumentParser(description="Summarise a document")
    subparsers = parser.add_subparsers(dest='command', help="Sub-command to execute.")
    
    # Add 'summarise' subcommand
    summarise_parser = subparsers.add_parser('summarise', help="Summarise a document")
    summarise_parser.add_argument('filename', help="The name of the file to summarise")
    
    # Check if no arguments or unrecognized argument is passed
    if len(sys.argv) == 1:
        parser.print_help()
    elif len(sys.argv) == 2 and sys.argv[1] not in ['summarise']:
        args = parser.parse_args(['summarise', sys.argv[1]])
        _summarise(args.filename)
    else:
        args = parser.parse_args()
        if args.command == 'summarise':
            _summarise(args.filename)
        else:
            print('It looks like you need help')
            parser.print_help()

def _summarise(filename):
    if filename.endswith('.md') or filename.endswith('.txt'):
        summarise_file(filename)
    elif filename.endswith('.pdf'):
        summarise_pdf(filename)
    elif filename.startswith('http://') or filename.startswith('https://'):
        summarise_web(filename)
    else:
        print("File not recognised")

