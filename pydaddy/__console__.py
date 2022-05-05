import argparse
import pkg_resources
import subprocess
import os

banner = """    ____        ____            __    __     
   / __ \__  __/ __ \____ _____/ /___/ /_  __
  / /_/ / / / / / / / __ `/ __  / __  / / / /
 / ____/ /_/ / /_/ / /_/ / /_/ / /_/ / /_/ / 
/_/    \__, /_____/\__,_/\__,_/\__,_/\__, /  
      /____/                        /____/   
              _       ___ _  _               
             |_)       | |_ |_ __ |   _. |_  
             |_) \/    | |_ |_    |_ (_| |_) 
                 /                           
"""

parser = argparse.ArgumentParser()

parser.add_argument('file',
                    metavar='path',
                    type=str,
                    help='data file to be analysed')
parser.add_argument('--column_format',
                    type=str,
                    default='x',
                    help='column header of data file')
parser.add_argument('-t',
                    type=float,
                    default=1,
                    help='time step of the time series data')
parser.add_argument('--delimiter',
                    type=str,
                    default=',',
                    help='csv delimiter of data file')

def main():
    args = parser.parse_args()

    os.environ['pydaddy_data_file'] = args.file
    os.environ['pydaddy_data_col_fmt'] = args.column_format
    os.environ['pydaddy_data_delimiter'] = args.delimiter
    os.environ['pydaddy_t'] = str(args.t)

    report_nb = pkg_resources.resource_string('pydaddy', 'report/report')

    with open('sample_report.ipynb', 'w') as f:
        f.write(report_nb.decode())

    cmd = 'jupyter nbconvert --log-level ERROR --execute --TemplateExporter.exclude_input=True sample_report.ipynb --to html'

    subprocess.call("clear")
    print(banner)
    print("*Experimental feature*")
    os.system(cmd)
    os.remove('sample_report.ipynb')
    os.remove('data.pydaddy.csv')

if __name__ == "__main__":
    main()