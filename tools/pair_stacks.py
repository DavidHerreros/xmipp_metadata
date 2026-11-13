#!/usr/bin/env python
"""
Test whether a set of <tomo>/<n>_stack2d.mrcs can be paired with a particles STAR
that does not name them.

    python pair_stacks.py particles.star tomograms.star /root/holding/the/stacks

Stacks are indexed by (parent folder, leading number) and matched against
(rlnTomoName, the number in rlnImageName), so the folder depth does not matter.
Visibility is recomputed and compared against each stack's real slice count.
"""

import collections
import glob
import os
import sys
import warnings

import mrcfile

from xmipp_metadata.metadata.relion_tomo import _read_star, tomo_star_to_tilt_particles


def _stem(row, i):
    """The particle number the STAR already encodes in its image path."""
    base = os.path.basename(str(row.get("rlnImageName", "")).split("@")[-1])
    for suffix in ("_data.mrc", "_stack2d.mrcs", ".mrc", ".mrcs"):
        base = base.replace(suffix, "")
    return base or str(i + 1)


def main(particles_star, tomograms_star, stack_root):
    blocks = _read_star(particles_star)
    parts = blocks.get("particles", max(blocks.values(), key=lambda d: len(d.columns)))
    parts = parts.reset_index(drop=True)

    stacks = {}
    for f in glob.glob(os.path.join(stack_root, "**", "*_stack2d.mrcs"), recursive=True):
        stacks[(os.path.basename(os.path.dirname(f)),
                os.path.basename(f).split("_stack2d")[0])] = f
    print(f"{len(stacks)} stacks under {stack_root}")

    folders = {k[0] for k in stacks}
    tomos = set(parts["rlnTomoName"].astype(str)) if "rlnTomoName" in parts else set()
    print(f"folders: {len(folders)}   rlnTomoName values: {len(tomos)}   "
          f"overlap: {len(folders & tomos)}")
    if not folders & tomos:
        print(f"  folder example:      {sorted(folders)[:2]}")
        print(f"  rlnTomoName example: {sorted(tomos)[:2]}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = tomo_star_to_tilt_particles(particles_star, tomograms_star)
    computed = collections.Counter(df["subtomo_labels"].to_numpy())

    found = matched = 0
    misses = []
    for i in range(len(parts)):
        row = parts.iloc[i]
        key = (str(row.get("rlnTomoName", "")), _stem(row, i))
        path = stacks.get(key)
        if path is None:
            if len(misses) < 10:
                misses.append((key, "no such stack"))
            continue
        found += 1
        with mrcfile.mmap(path, permissive=True) as m:
            n = m.data.shape[0] if m.data.ndim == 3 else 1
        want = computed.get(i + 1, 0)
        if n == want:
            matched += 1
        elif len(misses) < 10:
            misses.append((key, f"slices={n} computed={want}"))

    print(f"\nparticles: {len(parts)}   stacks resolved: {found}   "
          f"slice counts matching: {matched}")
    for key, why in misses:
        print(f"  {key}  {why}")

    rate = matched / found if found else 0.0
    print()
    if found < len(parts) * 0.9:
        print(f"VERDICT: only {found}/{len(parts)} stacks resolve -- the naming or the "
              f"folder layout does not correspond. Do not pair these.")
    elif rate > 0.98:
        print(f"VERDICT: {rate:.1%} of slice counts agree -- the pairing is consistent.")
    elif rate > 0.7:
        print(f"VERDICT: {rate:.1%} agree. The pairing is probably right but the "
              f"visibility criterion differs from the one used at extraction. Usable "
              f"only if you keep the {matched} agreeing particles and drop the rest.")
    else:
        print(f"VERDICT: only {rate:.1%} agree -- these stacks are not these particles.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
