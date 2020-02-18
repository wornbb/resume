import re
from operator import attrgetter, itemgetter
class EReport():
    def parse(self, f_name):
        """parse the file and store the data.
        f_name: string of file name
        """
        self.employees = []
        with open(f_name, 'r') as f:
            for line in f: # loop through the file
                comment_location = line.find("#")
                if comment_location != -1:
                    # ignore all strings after the first #
                    line = line[:comment_location]
                if line: # if line is not empty
                    line = line[:-1] # stripe new_line
                    [no, first, last] = re.findall(r'\w+', line)
                    self.employees.append({'no': no, 'first': first, 'last': last})
    def _print_all(self):
        for employee in self.employees:
            print(employee['no'] + ',' + employee['first'] + ' ' + employee['last'])
        print()
    def print_sort_by_ln(self):
        print('Processing by employee number...')
        self.employees.sort(key=itemgetter('last'))
        self._print_all()
    def print_sort_by_no(self):
        print('Processing by last (family) Name..')
        self.employees.sort(key=itemgetter('no'))
        self._print_all()

if __name__ == "__main__":
    f_name = "employees.dat"
    parser = EReport()
    parser.parse(f_name)
    parser.print_sort_by_no()
    parser.print_sort_by_ln()