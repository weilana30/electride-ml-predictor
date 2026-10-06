#!/usr/bin/env python3
"""Find the ELF maximum and sample the density at that location."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cube:
    shape: tuple[int, int, int]
    origin: tuple[float, float, float]
    grid_vectors: tuple[tuple[float, float, float], ...]
    values: list[float]

    @property
    def cell_vectors(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            tuple(component * count for component in vector)
            for vector, count in zip(self.grid_vectors, self.shape)
        )


def read_cube(path: Path) -> Cube:
    with path.open("rb") as handle:
        handle.readline()
        handle.readline()
        atom_line = handle.readline().split()
        n_atoms = abs(int(atom_line[0]))
        origin = tuple(float(value) for value in atom_line[1:4])

        shape = []
        grid_vectors = []
        for _ in range(3):
            grid_line = handle.readline().split()
            shape.append(abs(int(grid_line[0])))
            grid_vectors.append([float(value) for value in grid_line[1:4]])

        for _ in range(n_atoms):
            handle.readline()

        values = [float(value) for value in handle.read().split()]

    expected = shape[0] * shape[1] * shape[2]
    if len(values) != expected:
        raise ValueError(
            f"{path}: expected {expected} grid values, found {len(values)}"
        )

    return Cube(
        shape=tuple(shape),
        origin=origin,
        grid_vectors=tuple(tuple(vector) for vector in grid_vectors),
        values=values,
    )


def grid_value(cube: Cube, index: tuple[int, int, int]) -> float:
    i, j, k = index
    nx, ny, _ = cube.shape
    return cube.values[i + nx * (j + ny * k)]


def cartesian_position(cube: Cube, index: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(
        cube.origin[component]
        + sum(index[axis] * cube.grid_vectors[axis][component] for axis in range(3))
        for component in range(3)
    )


def solve_3x3(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    a, b, c = matrix
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(determinant) < 1e-14:
        raise ValueError("Cube cell vectors are singular")

    inverse = (
        (
            (b[1] * c[2] - b[2] * c[1]) / determinant,
            (a[2] * c[1] - a[1] * c[2]) / determinant,
            (a[1] * b[2] - a[2] * b[1]) / determinant,
        ),
        (
            (b[2] * c[0] - b[0] * c[2]) / determinant,
            (a[0] * c[2] - a[2] * c[0]) / determinant,
            (a[2] * b[0] - a[0] * b[2]) / determinant,
        ),
        (
            (b[0] * c[1] - b[1] * c[0]) / determinant,
            (a[1] * c[0] - a[0] * c[1]) / determinant,
            (a[0] * b[1] - a[1] * b[0]) / determinant,
        ),
    )
    return tuple(sum(row[i] * vector[i] for i in range(3)) for row in inverse)


def fractional_position(
    cube: Cube, cartesian: tuple[float, float, float]
) -> tuple[float, float, float]:
    relative = tuple(cartesian[i] - cube.origin[i] for i in range(3))
    transposed_cell = tuple(
        tuple(cube.cell_vectors[j][i] for j in range(3)) for i in range(3)
    )
    return solve_3x3(transposed_cell, relative)


def interpolate_periodic(cube: Cube, cartesian: tuple[float, float, float]) -> float:
    fractional = fractional_position(cube, cartesian)
    coordinate = tuple(
        (fractional[i] * cube.shape[i]) % cube.shape[i] for i in range(3)
    )
    lower = tuple(int(value) for value in coordinate)
    upper = tuple((lower[i] + 1) % cube.shape[i] for i in range(3))
    weight = tuple(coordinate[i] - lower[i] for i in range(3))

    value = 0.0
    for ix, wx in ((lower[0], 1.0 - weight[0]), (upper[0], weight[0])):
        for iy, wy in ((lower[1], 1.0 - weight[1]), (upper[1], weight[1])):
            for iz, wz in ((lower[2], 1.0 - weight[2]), (upper[2], weight[2])):
                value += wx * wy * wz * grid_value(cube, (ix, iy, iz))
    return float(value)


def analyze(elf_path: Path, density_path: Path) -> dict[str, float | str]:
    elf = read_cube(elf_path)
    density = read_cube(density_path)
    if any(
        abs(elf.cell_vectors[i][j] - density.cell_vectors[i][j]) > 1e-8
        for i in range(3)
        for j in range(3)
    ):
        raise ValueError("ELF and density cubes do not describe the same cell")

    flat_index = max(range(len(elf.values)), key=elf.values.__getitem__)
    nx, ny, _ = elf.shape
    maximum_index = (
        flat_index % nx,
        (flat_index // nx) % ny,
        flat_index // (nx * ny),
    )
    maximum_position = cartesian_position(elf, maximum_index)
    fractional = fractional_position(elf, maximum_position)
    density_at_maximum = interpolate_periodic(density, maximum_position)

    return {
        "elf_cube": str(elf_path),
        "density_cube": str(density_path),
        "elf_max": grid_value(elf, maximum_index),
        "density_at_elf_max": density_at_maximum,
        "grid_i": int(maximum_index[0]),
        "grid_j": int(maximum_index[1]),
        "grid_k": int(maximum_index[2]),
        "x": float(maximum_position[0]),
        "y": float(maximum_position[1]),
        "z": float(maximum_position[2]),
        "frac_x": float(fractional[0]),
        "frac_y": float(fractional[1]),
        "frac_z": float(fractional[2]),
    }


def write_result(path: Path, result: dict[str, float | str], label: str | None) -> None:
    row = {"label": label or "", **result}
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("elf_cube", type=Path)
    parser.add_argument("density_cube", type=Path)
    parser.add_argument("--output", type=Path, help="Append the result to a CSV file")
    parser.add_argument("--label", help="Label such as low, base, or high")
    args = parser.parse_args()

    result = analyze(args.elf_cube, args.density_cube)
    if args.output:
        write_result(args.output, result, args.label)
    else:
        print(",".join(result.keys()))
        print(",".join(str(value) for value in result.values()))


if __name__ == "__main__":
    main()
