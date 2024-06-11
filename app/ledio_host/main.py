import argparse
import sys
from ledio_host.commands import chat, save_html, save_pdf

def main():
    parser = argparse.ArgumentParser(description="Ledio Host - your radio host")
    subparsers = parser.add_subparsers(dest='command', help="Sub-command to execute.")

    # Launch chat interface if no arguments are provided
    if len(sys.argv) == 1:
        chat()
    else:
        args = parser.parse_args()
        if args.command == 'save':
            _save(args.destination)
        else:
            print('Unknown command. Use -h for help.')
            parser.print_help()

def _save(destination):
    if not destination:
        print("Reading from stdin (not yet implemented)")
    elif destination.startswith('http://') or destination.startswith('https://'):
        print(f"Saving URL: {destination} (not yet implemented)")
    elif destination.endswith('.html'):
        save_html(destination)
    elif destination.endswith('.pdf'):
        save_pdf(destination)
    else:
        print(f"Saving file: {destination} (not yet implemented)")
