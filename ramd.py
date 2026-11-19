"""
Random Acceleration Molecular Dynamics (RAMD).

A constant-magnitude force of randomly chosen orientation is applied to the
centre of mass of the ligand. Ligand progress is evaluated every `ramd_steps`
steps: if the COM has travelled less than `r_min` in that block, a new random
direction is drawn, otherwise the current direction is retained. This drives
egress from buried binding sites without presupposing an exit route.

Reference: Luedemann, Lounnas & Wade, J. Mol. Biol. 303:797 (2000), developed
on cytochrome P450cam. Defaults follow the NAMD 3.0 implementation
(force 16 kcal/mol/A, ramd_steps 50, r_min 0.01 A).
"""

import numpy as np

import openmm
import openmm.unit as unit

# NAMD default force, expressed in OpenMM units (kJ/mol/nm).
KCAL_PER_MOL_ANGSTROM = 4.184 / 0.1
DEFAULT_FORCE = 16.0 * KCAL_PER_MOL_ANGSTROM


def add_ramd_force(system, ligand_indices, force_group=30):
    """Attach the RAMD COM force to `system`.

    The force is expressed on the mass-weighted centroid of the ligand, so the
    resulting acceleration is distributed over the ligand atoms exactly as a
    force applied to its centre of mass. Direction is carried in global
    parameters so it can be updated each block without rebuilding the context.

    Must be called before the Context is created. Returns the Force.
    """
    force = openmm.CustomCentroidBondForce(1, "-(fx*x1 + fy*y1 + fz*z1)")
    for name in ("fx", "fy", "fz"):
        force.addGlobalParameter(name, 0.0)
    group = force.addGroup([int(i) for i in ligand_indices])
    force.addBond([group], [])
    force.setForceGroup(force_group)
    system.addForce(force)
    return force


def random_unit_vector(rng):
    """Draw a direction uniformly distributed on the unit sphere."""
    while True:
        v = rng.normal(size=3)
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            return v / norm


class RAMDSimulation:
    """Drive an OpenMM Simulation under RAMD until the ligand dissociates.

    The Simulation must already contain the force returned by
    `add_ramd_force`, built over the same `ligand_indices` passed here.
    """

    def __init__(self, simulation, ligand_indices, protein_indices,
                 force_magnitude=DEFAULT_FORCE, ramd_steps=50,
                 r_min=0.001, dissociation_distance=2.0, seed=None,
                 max_block_displacement=0.5):
        # r_min, dissociation_distance and max_block_displacement are in nm
        # (0.001 nm = 0.01 A). Thermal plus driven motion of a ligand over a
        # 50-step block is of order 0.01-0.05 nm, so 0.5 nm signals a blow-up.
        self.simulation = simulation
        self.max_block_displacement = max_block_displacement
        self.ligand = np.asarray(ligand_indices, dtype=int)
        self.protein = np.asarray(protein_indices, dtype=int)
        self.protein_coarse = self.protein[::10]
        self.force_magnitude = force_magnitude
        self.ramd_steps = ramd_steps
        self.r_min = r_min
        self.dissociation_distance = dissociation_distance
        self.rng = np.random.default_rng(seed)

        masses = [simulation.system.getParticleMass(int(i)).value_in_unit(unit.dalton)
                  for i in self.ligand]
        self.ligand_masses = np.asarray(masses)
        self.total_mass = self.ligand_masses.sum()

        self.direction_changes = 0
        self.blocks_run = 0

    def _state(self):
        # One synchronisation per block: pulling positions and energy from a
        # single State avoids a second GPU round trip in the inner loop.
        # enforcePeriodicBox=False keeps the ligand's path continuous as it
        # leaves the protein, which wrapping would otherwise fragment.
        state = self.simulation.context.getState(
            getPositions=True, getEnergy=True, enforcePeriodicBox=False)
        positions = state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer)
        energy = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        return positions, energy

    def _check_stable(self, positions, energy, travelled, steps_done):
        """Guard against an unstable run being mistaken for dissociation.

        A blown-up system sends interatomic distances to arbitrarily large
        values, which the separation test would otherwise read as a successful
        egress event. Fail loudly instead.
        """
        if not np.isfinite(energy) or not np.isfinite(positions[self.ligand]).all():
            raise RuntimeError(
                "Simulation became unstable at step %d (potential energy %s). "
                "Minimise and equilibrate before applying the RAMD force."
                % (steps_done, energy))
        if travelled > self.max_block_displacement:
            raise RuntimeError(
                "Ligand centre of mass moved %.2f nm in one %d-step block at "
                "step %d, which is not physical; the system has most likely "
                "blown up rather than dissociated. Minimise and equilibrate "
                "first, or lower the force."
                % (travelled, self.ramd_steps, steps_done))

    def _ligand_com(self, positions):
        return (positions[self.ligand] * self.ligand_masses[:, None]).sum(0) / self.total_mass

    def _pairwise_min(self, lig, protein_indices, positions):
        # Chunked to keep the pairwise block small for large receptors.
        best = np.inf
        for start in range(0, len(protein_indices), 2048):
            prot = positions[protein_indices[start:start + 2048]]
            d = np.linalg.norm(lig[:, None, :] - prot[None, :, :], axis=-1)
            best = min(best, d.min())
        return best

    def _min_ligand_protein_distance(self, positions):
        """Closest ligand-receptor approach, in nm.

        The full ligand-by-receptor comparison is the most expensive step in
        the block, and while the ligand is bound its answer is never in doubt.
        A decimated subset of the receptor gives an upper bound on the true
        minimum, so a subset distance already inside the dissociation
        criterion settles the question without the full calculation.
        """
        lig = positions[self.ligand]
        coarse = self._pairwise_min(lig, self.protein_coarse, positions)
        if coarse <= self.dissociation_distance:
            return coarse
        return self._pairwise_min(lig, self.protein, positions)

    def _set_direction(self, direction):
        scaled = direction * self.force_magnitude
        for name, value in zip(("fx", "fy", "fz"), scaled):
            self.simulation.context.setParameter(name, float(value))

    def run(self, max_steps, callback=None):
        """Run until the ligand dissociates or `max_steps` is reached.

        Returns a dict describing the outcome; `exit_step` is None if the
        ligand never dissociated.
        """
        direction = random_unit_vector(self.rng)
        self._set_direction(direction)

        positions, _ = self._state()
        previous_com = self._ligand_com(positions)
        steps_done = 0

        while steps_done < max_steps:
            block = min(self.ramd_steps, max_steps - steps_done)
            self.simulation.step(block)
            steps_done += block
            self.blocks_run += 1

            positions, energy = self._state()
            com = self._ligand_com(positions)
            travelled = np.linalg.norm(com - previous_com)
            previous_com = com

            self._check_stable(positions, energy, travelled, steps_done)

            separation = self._min_ligand_protein_distance(positions)
            if callback is not None:
                callback(steps_done, travelled, separation)

            if separation > self.dissociation_distance:
                return {"dissociated": True, "exit_step": steps_done,
                        "separation": separation,
                        "direction_changes": self.direction_changes,
                        "blocks": self.blocks_run}

            # No appreciable progress this block: try a different direction.
            if travelled < self.r_min:
                direction = random_unit_vector(self.rng)
                self._set_direction(direction)
                self.direction_changes += 1

        return {"dissociated": False, "exit_step": None,
                "separation": separation,
                "direction_changes": self.direction_changes,
                "blocks": self.blocks_run}
