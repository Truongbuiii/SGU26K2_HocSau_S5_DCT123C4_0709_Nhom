"""
Vai tro: TV3 - Thuc nghiem va danh gia
File   : run_experiments.py - chay hang loat theo ExpID, bo qua run da xong, ghi nhat ky

Vi du:
    python run_experiments.py --ids E2-01 E2-02 --runner "Nguyen Van A"
    python run_experiments.py --group E2 --runner "Nguyen Van A"
    python run_experiments.py --ids E2-01 --smoke           # chay thu 1 epoch, ghi vao results/smoke
    python run_experiments.py --ids E2-03 --force           # chay lai: thu muc cu duoc DOI TEN, khong xoa
"""
import argparse
import os
import time
import traceback
from dataclasses import replace
from datetime import datetime

from config import get_config, group_ids, run_dir_path


def is_done(run_dir):
    """Run xong khi co best.pt va last_ckpt.pt da bi train.py xoa (train.py xoa no khi hoan tat)."""
    return (all(os.path.exists(os.path.join(run_dir, f)) for f in ("config.json", "history.csv", "best.pt"))
            and not os.path.exists(os.path.join(run_dir, "last_ckpt.pt")))


def can_resume(run_dir):
    """Run dang do (Kaggle/Colab ngat phien): train.py tu noi tiep tu last_ckpt.pt."""
    return os.path.exists(os.path.join(run_dir, "last_ckpt.pt"))


def now():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--group")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--runner", default="", help="ten nguoi chay (ghi vao config.json)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, ghi vao results/smoke, khong ghi nhat ky")
    a = ap.parse_args()
    ids = list(a.ids or []) + (group_ids(a.group) if a.group else [])
    if not ids:
        ap.error("can --ids hoac --group")

    failed = []
    for exp_id in ids:
        cfg = get_config(exp_id, seed=a.seed)
        cfg = replace(cfg, runner=a.runner)
        if a.smoke:
            cfg = replace(cfg, epochs=1, output_root="results/smoke")
        run_dir = run_dir_path(cfg)

        if os.path.isdir(run_dir) and is_done(run_dir) and not a.force:
            if os.path.exists(os.path.join(run_dir, "test_metrics.json")):
                print(f"[skip] {run_dir} da xong. Dung --force de chay lai.")
            elif not a.no_eval:
                print(f"[eval-only] {run_dir} da train xong nhung chua danh gia -> chi chay evaluate")
                try:
                    from evaluate import evaluate
                    evaluate(run_dir)
                    if not a.smoke:
                        from aggregate import append_log
                        append_log(run_dir)
                except Exception:
                    traceback.print_exc()
                    failed.append(exp_id)
            continue
        if os.path.isdir(run_dir) and can_resume(run_dir) and not a.force:
            print(f"[resume] {run_dir} dang do, noi tiep tu last_ckpt.pt")
        elif os.path.isdir(run_dir):                       # chay lai (--force) hoac run hong: giu lai, doi ten
            os.rename(run_dir, f"{run_dir}_failed_{int(time.time())}")

        print("=" * 78)
        print(f"BAT DAU {cfg.exp_id} {cfg.exp_name} seed={cfg.seed} | nguoi chay: {cfg.runner} | {now()}")
        print(cfg)
        print("=" * 78, flush=True)
        t0 = time.time()
        try:
            from train import train                        # TV2
            rd = str(train(cfg))        # train() tra ve Path
            if not a.no_eval:
                from evaluate import evaluate
                evaluate(rd)
            if not a.smoke:
                from aggregate import append_log
                append_log(rd)
        except Exception:
            traceback.print_exc()
            failed.append(exp_id)
            print(f"[LOI] {exp_id} - thu muc run duoc giu lai, ghi ly do vao nhat ky roi chay lai.")
        print(f"KET THUC {exp_id} sau {(time.time() - t0) / 60:.1f} phut | {now()}\n", flush=True)

    print("Hoan tat." + (f" Run loi: {failed}" if failed else " Khong co loi."))


if __name__ == "__main__":
    main()
