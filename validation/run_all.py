#!/usr/bin/env python3
"""
Run all validation tests.

Usage::

    python -m vbrc_V2Tpy.validation.run_all --sweep sweep.npz
    python -m vbrc_V2Tpy.validation.run_all --sweep sweep.npz --method xfit_premelt
"""
import argparse

from .validate_prem import validate_prem
from .validate_roundtrip import validate_roundtrip


def main():
    parser = argparse.ArgumentParser(
        description='Run both PREM and round-trip validation cases',
    )
    parser.add_argument('--sweep', '-s', required=True,
                        help='Path to pre-computed sweep file')
    parser.add_argument('--method', '-m', default='xfit_premelt',
                        help='Anelastic method')
    parser.add_argument('--gs-prior', default='log_normal_1cm',
                        choices=['log_uniform', 'log_normal_1mm', 'log_normal_1cm'])
    parser.add_argument('--sigma-vs', type=float, default=0.05)
    parser.add_argument('--output', '-o', default='validation_results',
                        help='Base output directory')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    v = not args.quiet

    print("=" * 60)
    print("VALIDATION CASE 1: PREM Velocity Inversion")
    print("=" * 60)
    prem_results = validate_prem(
        args.sweep,
        anelastic_method=args.method,
        gs_prior_case=args.gs_prior,
        sigma_vs=args.sigma_vs,
        output_dir=f'{args.output}/prem',
        show=args.show,
        verbose=v,
    )

    print()
    print("=" * 60)
    print("VALIDATION CASE 2: Round-Trip (adiabat, no melt)")
    print("=" * 60)
    rt_results = validate_roundtrip(
        anelastic_method=args.method,
        gs_prior_case=args.gs_prior,
        sigma_vs=args.sigma_vs,
        output_dir=f'{args.output}/roundtrip',
        show=args.show,
        verbose=v,
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Round-trip RMSE(T) = {rt_results['rmse_T']:.1f} °C  "
          f"MAE(T) = {rt_results['mae_T']:.1f} °C")
    print(f"Outputs in: {args.output}/")


if __name__ == '__main__':
    main()
