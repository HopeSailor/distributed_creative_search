"""Run one or all analysis stages. Research inputs stay in a private directory."""
import argparse
from pathlib import Path
import study1
import study2
import diagnostics
import validation

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['study1','study2','diagnostics','validation','all'])
    parser.add_argument('--inputs',type=Path,required=True,help='Private analysis-ready input directory')
    parser.add_argument('--outputs',type=Path,required=True,help='Local generated results directory')
    args=parser.parse_args()
    source=args.inputs.resolve(); destination=args.outputs.resolve()
    if source==destination or source in destination.parents or destination in source.parents:
        parser.error('Input and output directories must be separate and non-nested.')
    if not source.is_dir(): parser.error('Input directory does not exist. No research data are distributed.')
    if args.stage in ('study1','all'): study1.run(source,destination/'study1')
    cells=None
    if args.stage in ('study2','all'): cells=study2.run(source,destination/'study2')
    if args.stage in ('diagnostics','all'):
        if cells is None: cells=study2.build_cells(*study2.load_inputs(source))
        diagnostics.run(cells,destination/'diagnostics')
    if args.stage in ('validation','all'): validation.run(source,destination/'validation')
    print('Completed. Results are local and are not uploaded by this program.')

if __name__=='__main__': main()

