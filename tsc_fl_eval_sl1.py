# tsc_fl_eval_sl1.py — convenience wrapper for the SmoothL1 long-run experiment.
#
# Equivalent CLI invocation:
#   python tsc_fl_eval.py --reg_types sl1 --n_rounds 20 --no_cache --exp_suffix long_run
#
# Keep this file if you prefer a zero-argument entry point for the SL1 run.
# All logic lives in tsc_fl_eval.py → main().

import argparse

import config
from tsc_fl_eval import main

if __name__ == "__main__":
    # Single optional flag so the SL1 long-run can be GPU-accelerated while
    # staying an effectively zero-argument entry point.
    p = argparse.ArgumentParser(description="SmoothL1 long-run FL experiment")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=config.USE_GPU,
                   help="Use GPU (CuPy) for the readout math; falls back to CPU "
                        "automatically if CuPy/CUDA is unavailable")
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false",
                   help="Force CPU even if the project default enables GPU")
    args = p.parse_args()

    main(
        reg_type        = "sl1",
        n_rounds        = 100,
        use_cache       = True,   # always reload from disk for the long run
        use_gpu         = args.use_gpu,
        exp_suffix      = "test_run",
        local_lr        = 1.0,
        global_lr       = 1.0,
        use_line_search = True,
        patience        = 5,
        save_model      = True,   # deliberate final run — persist the model
    )
