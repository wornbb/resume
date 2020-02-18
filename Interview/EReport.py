import re
from operator import itemgetter
import argparse
class EReport():
    def parse(self, f_name):
        """parse the file and store the data.
        f_name: string of file name
        """
        self.employees = []
        with open(f_name, 'r') as f:
            for line in f: # loop through the file
                line = line.rstrip('\n') # stripe new_line
                comment_location = line.find("#")
                if comment_location != -1:
                    # ignore all strings after the first #
                    line = line[:comment_location]
                if line: # if line is not empty
                    # it runs at python 3, so all characters from all language should be matched
                    [no, first, last] = re.findall(r'\w+', line)
                    # assuming all inputs are valid, because error handling is not specified.
                    self.employees.append({'no': no, 'first': first.capitalize(), 'last': last.capitalize()})
    def _print_all(self):
        for employee in self.employees:
            print(employee['no'] + ',' + employee['first'] + ' ' + employee['last'])
        print()
    def print_sort_by_ln(self):
        print('Processing by last (family) Name...')
        self.employees.sort(key=itemgetter('last', 'no'))
        self._print_all()
    def print_sort_by_no(self):
        print('Processing by employee number...')
        self.employees.sort(key=itemgetter('no'))
        self._print_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('f_name', metavar='I', type=str, nargs='?', default='employees.dat',
                    help='Give me a path or I will guess')
    args = parser.parse_args()
    report = EReport()
    report.parse(args.f_name)
    report.print_sort_by_no()
    report.print_sort_by_ln()