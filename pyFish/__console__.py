import argparse
import numpy as np

from pyFish.__main__ import Characterize

parser = argparse.ArgumentParser()

parser.add_argument('file',
                    metavar='path',
                    type=str,
                    help='data file to be analysed')
parser.add_argument('--delimiter',
                    type=str,
                    default=',',
                    help='csv delimiter of input file')
parser.add_argument('--vector', type=bool, nargs='?', const=True)
parser.add_argument('-x1',
                    type=int,
                    default=0,
                    help='index of first data column')
parser.add_argument('-x2',
                    type=int,
                    default=1,
                    help='index of second data column')
parser.add_argument('-t', type=int, help='index of time stamp data')
parser.add_argument('-t_int', type=float, help='time increment')
parser.add_argument('-dt', type=int, help='dt')
parser.add_argument('-delta_t', type=int, default=1, help='delta_t')
parser.add_argument('-t_lag', type=int, default=1000, help='t_lag')
parser.add_argument('-inc', type=float, default=0.01, help='inc')
parser.add_argument('-inc_x', type=float, default=0.1, help='inc_x')
parser.add_argument('-inc_y', type=float, default=0.1, help='inc_y')
parser.add_argument('-max_order', type=int, default=10, help='max_order')
parser.add_argument('--no_fft',
                    type=bool,
                    nargs='?',
                    const=False,
                    help="use standard method instead of fft for analysis")
parser.add_argument('--drift_order',
                    type=int,
                    help='force drift order to a value')
parser.add_argument('--diff_order',
                    type=int,
                    help='force diff order to a value')
parser.add_argument('-n_trials', type=int, default=1, help='n_trials')
parser.add_argument('--show_figs',
                    type=bool,
                    nargs='?',
                    const=True,
                    help='Show figures')
parser.add_argument('--savepath',
                    metavar='path',
                    type=str,
                    help='path to save the results')


def main():
    args = parser.parse_args()
    data = np.loadtxt(args.file, delimiter=args.delimiter)
    t = data[:, args.t] if args.t is not None else None

    if args.vector:
        x1 = data[:, args.x1]
        x2 = data[:, args.x2]
        d = [x1, x2]
    else:
        d = [data[:, args.x1]]

    dt = 'auto' if args.dt is None else args.dt
    fft = True if args.no_fft is None else False
    show = False if args.show_figs is None else True

    out = Characterize(
        data=d,
        t=t,
        t_int=args.t_int,
        dt=dt,
        delta_t=args.delta_t,
        t_lag=args.t_lag,
        inc=args.inc,
        inc_x=args.inc_x,
        inc_y=args.inc_y,
        max_order=args.max_order,
        fft=fft,
        drift_order=args.drift_order,
        diff_order=args.diff_order,
        order_metric="R2_adj",
        simple_method=True,
        n_trials=args.n_trials,
    )

    out.save_all_data(show=show, savepath=args.savepath)


if __name__ == '__main__':
    main()
