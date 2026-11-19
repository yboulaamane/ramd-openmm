# ramdRunner: RAMD ligand egress on OpenMM

`ramdRunner` runs Random Acceleration Molecular Dynamics (RAMD) to drive a
ligand out of a buried binding site, for systems where ligand-selective
boosting (LiGaMD) does not produce dissociation. A randomly oriented force
acts on the ligand's centre of mass. If the ligand moves less than `--r-min`
over a block of `--ramd-steps`, the direction is redrawn. The replica stops
when the ligand is `--dissociation` Å from the protein, or at `--max-steps`.

Each replica is an independent trajectory. The distribution of exit times over
many replicas is the basis of a tau-RAMD residence-time estimate. The exit
routes show the egress channels.

Two files: the RAMD force and driver are in `ramd.py` (`add_ramd_force`,
`RAMDSimulation`), and the command-line runner is `ramdRunner`. The LiGaMD /
GaMD code lives separately in
[ligamd-openmm](https://github.com/yboulaamane/ligamd-openmm).

## Requirements

OpenMM (tested with 8.2, CUDA platform), numpy and mdtraj. Keep `ramd.py`
next to `ramdRunner`: Python puts the script's folder on the import path, so
`python /path/to/ramdRunner ...` works from any directory.

## Usage

```
usage: ramdRunner [-h] -l LIGAND [-f FORCE] [-n REPLICAS] [-m MAX_STEPS]
                  [--ramd-steps RAMD_STEPS] [--r-min R_MIN] [-d DISSOCIATION]
                  [-p PLATFORM] [--device-index DEVICE_INDEX] [-o OUTPUT]
                  [--no-minimize] [--minimize-steps MINIMIZE_STEPS]
                  [--equilibrate EQUILIBRATE] [--seed SEED]
                  [--temperature TEMPERATURE] [--timestep TIMESTEP]
                  [--friction FRICTION] [--cutoff CUTOFF] [--full-system]
                  [--trajectory-interval TRAJ_INTERVAL]
                  PRMTOP INPCRD

positional arguments:
  PRMTOP                Amber topology file.
  INPCRD                Amber coordinate file.

options:
  -l, --ligand          Residue name of the ligand, e.g. 'ERY'.
  -f, --force           RAMD force magnitude in kcal/mol/Angstrom. Default
                        16.0, following NAMD. Run several magnitudes to
                        confirm the egress route is not an artefact of the
                        applied force.
  -n, --replicas        Number of independent RAMD replicas.
  -m, --max-steps       Step ceiling per replica (default 5 ns).
  --ramd-steps          Steps per RAMD block between evaluations.
  --r-min               Minimum COM displacement per block, in Angstroms,
                        below which the force direction is redrawn.
                        Default 0.01.
  -d, --dissociation    Ligand-protein separation in Angstroms at which the
                        ligand is counted as dissociated. Must be smaller
                        than the solvent buffer, or the ligand meets its own
                        periodic image before the criterion can be met.
  -p, --platform        OpenMM platform.
  --device-index        CUDA device index.
  -o, --output          Output directory.
  --no-minimize         Skip energy minimisation (on by default: applying the
                        RAMD force to an unrelaxed structure destabilises the
                        system).
  --minimize-steps      Cap on minimisation iterations.
  --equilibrate         Unbiased equilibration steps before RAMD.
  --seed                Base random seed.
  --temperature         Temperature in kelvin.
  --timestep            Timestep in picoseconds.
  --friction            Langevin friction in inverse picoseconds.
  --cutoff              Nonbonded cutoff in nanometers.
  --full-system         Write every atom to the trajectory, water included.
                        By default only solute atoms are written.
  --trajectory-interval Steps between trajectory frames.
```

## Example

One replica per job, as used for the CYP3A4 ligand-egress campaigns. With a
2 fs timestep, this is 1 ns of unbiased equilibration, then up to 10 ns of
RAMD at 10 kcal/mol/Å, with a frame every 1 ps:

```bash
python ramdRunner system.prmtop system_equil.rst7 \
    -l LIG -f 10.0 -n 1 -m 5000000 -d 8.0 \
    --equilibrate 500000 --seed 11001 \
    --trajectory-interval 500 -p CUDA -o ramd_out
```

Run many seeds, e.g. as a SLURM array with `--seed $((11000 + SLURM_ARRAY_TASK_ID))`.

## Outputs (in `-o`)

- `ramd_summary.txt`: one row per replica (dissociated, exit time in ns,
  number of direction changes), plus the egress count, mean and median.
- `replica<N>.dcd`: trajectory of the RAMD stage. By default it holds solute
  atoms only.
- `solute_reference.pdb`: topology and reference coordinates for loading the
  solute-only DCDs (not written with `--full-system`).

The log prints one line per replica, for example
`replica 1: EGRESS at 0.354 ns (176800 steps), 3 direction changes, 107 s wall`.
