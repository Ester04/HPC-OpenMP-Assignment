#!/usr/bin/env python3
"""Generate dependency-free SVG performance plots from perf_valgrind.sh CSV output."""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


VERSION_ORDER = ("v0", "v2", "v3", "v4")
COLORS = {
    "v0": "#4c566a",
    "v2": "#5e81ac",
    "v3": "#a3be8c",
    "v4": "#d08770",
}
WIDTH = 920
HEIGHT = 540
LEFT = 90
RIGHT = 240
TOP = 58
BOTTOM = 92


def parse_csv(csv_path):
    """Return successful perf times and Cachegrind instruction counts."""
    perf = defaultdict(list)
    cachegrind = defaultdict(list)
    required = {
        "tool",
        "version",
        "n",
        "threads",
        "rep",
        "wall_seconds",
        "vg_ir",
        "exit_status",
    }

    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV privo delle colonne richieste: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            if None in row:
                print(f"Avviso: riga {line_number} con colonne in eccesso, ignorata.", file=sys.stderr)
                continue
            try:
                n = int(row["n"])
                threads = int(row["threads"])
                if n < 1 or threads < 1:
                    raise ValueError
            except (TypeError, ValueError):
                print(f"Avviso: dimensione o thread non validi alla riga {line_number}, ignorata.", file=sys.stderr)
                continue

            key = (n, row["version"], threads)
            if row["tool"] == "perf" and row["exit_status"] == "0":
                try:
                    seconds = float(row["wall_seconds"])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(seconds) and seconds > 0:
                    perf[key].append(seconds)
            elif row["tool"] == "valgrind" and row["exit_status"] == "0":
                try:
                    instructions = int(row["vg_ir"])
                except (TypeError, ValueError):
                    continue
                if instructions > 0:
                    cachegrind[key].append(instructions)

    return perf, cachegrind


def ordered_versions(keys):
    found = {version for _, version, _ in keys}
    return [version for version in VERSION_ORDER if version in found] + sorted(
        found.difference(VERSION_ORDER)
    )


def svg_plot(
    path,
    title,
    y_label,
    x_values,
    series,
    y_max,
    scale=1,
    y_suffix="",
    error_ranges=None,
    x_label="Thread",
    log_y_min=None,
):
    """Write a line chart with categorical, evenly spaced x positions."""
    plot_width = WIDTH - LEFT - RIGHT
    plot_height = HEIGHT - TOP - BOTTOM
    x_positions = {
        value: LEFT + plot_width * (index / max(1, len(x_values) - 1))
        for index, value in enumerate(x_values)
    }
    y_max = y_max or 1
    if log_y_min is not None:
        log_min = math.floor(math.log10(log_y_min))
        log_max = max(log_min + 1, math.ceil(math.log10(y_max)))

        def y_position(value):
            return TOP + plot_height * (log_max - math.log10(value)) / (log_max - log_min)

        ticks = [10.0 ** exponent for exponent in range(log_min, log_max + 1)]
    else:
        def y_position(value):
            return TOP + plot_height * (1 - value / y_max)

        ticks = [y_max * tick / 5 for tick in range(6)]
    colors = [COLORS.get(name.split()[0], "#b48ead") for name, _ in series]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{WIDTH / 2}" y="28" text-anchor="middle" font-family="sans-serif" '
        f'font-size="19" font-weight="bold">{escape(title)}</text>',
    ]

    for value in ticks:
        y = y_position(value)
        parts.append(
            f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + plot_width}" y2="{y:.1f}" '
            'stroke="#e5e9f0"/>'
        )
        parts.append(
            f'<text x="{LEFT - 12}" y="{y + 5:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{value / scale:.3g}{y_suffix}</text>'
        )

    parts.extend(
        [
            f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + plot_height}" stroke="#4c566a"/>',
            f'<line x1="{LEFT}" y1="{TOP + plot_height}" x2="{LEFT + plot_width}" '
            f'y2="{TOP + plot_height}" stroke="#4c566a"/>',
        ]
    )
    for value, x in x_positions.items():
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{TOP + plot_height}" x2="{x:.1f}" '
                f'y2="{TOP + plot_height + 5}" stroke="#4c566a"/>',
                f'<text x="{x:.1f}" y="{TOP + plot_height + 23}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12">{value}</text>',
            ]
        )

    parts.extend(
        [
            f'<text x="{LEFT + plot_width / 2:.1f}" y="{HEIGHT - 24}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14">{escape(x_label)}</text>',
            f'<text x="22" y="{TOP + plot_height / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 22 {TOP + plot_height / 2:.1f})" '
            f'font-family="sans-serif" font-size="14">{escape(y_label)}</text>',
        ]
    )

    for series_index, (name, values) in enumerate(series):
        color = colors[series_index]
        points = []
        for thread in x_values:
            value = values.get(thread)
            if value is None:
                points.append(None)
                continue
            x = x_positions[thread]
            y = y_position(value)
            points.append((x, y))
        for left_point, right_point in zip(points, points[1:]):
            if left_point is not None and right_point is not None:
                parts.append(
                    f'<line x1="{left_point[0]:.1f}" y1="{left_point[1]:.1f}" '
                    f'x2="{right_point[0]:.1f}" y2="{right_point[1]:.1f}" '
                    f'stroke="{color}" stroke-width="2.5"/>'
                )
        for thread, point in zip(x_values, points):
            if point is None:
                continue
            x, y = point
            limits = (error_ranges or {}).get(name, {}).get(thread)
            if limits is not None:
                low_y = y_position(limits[0])
                high_y = y_position(limits[1])
                parts.extend(
                    [
                        f'<line x1="{x:.1f}" y1="{low_y:.1f}" x2="{x:.1f}" '
                        f'y2="{high_y:.1f}" stroke="{color}" stroke-width="1.5"/>',
                        f'<line x1="{x - 4:.1f}" y1="{low_y:.1f}" x2="{x + 4:.1f}" '
                        f'y2="{low_y:.1f}" stroke="{color}" stroke-width="1.5"/>',
                        f'<line x1="{x - 4:.1f}" y1="{high_y:.1f}" x2="{x + 4:.1f}" '
                        f'y2="{high_y:.1f}" stroke="{color}" stroke-width="1.5"/>',
                    ]
                )
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}">'
                f'<title>{escape(name)}, {escape(x_label)}={thread}: {values[thread] / scale:.5g}'
                f'{y_suffix}</title></circle>'
            )

        legend_y = TOP + 22 + series_index * 26
        parts.extend(
            [
                f'<line x1="{LEFT + plot_width + 28}" y1="{legend_y}" '
                f'x2="{LEFT + plot_width + 52}" y2="{legend_y}" stroke="{color}" '
                'stroke-width="2.5"/>',
                f'<circle cx="{LEFT + plot_width + 40}" cy="{legend_y}" r="4" fill="{color}"/>',
                f'<text x="{LEFT + plot_width + 62}" y="{legend_y + 4}" '
                f'font-family="sans-serif" font-size="13">{escape(name)}</text>',
            ]
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def median_by_thread(data, n, version):
    return {
        threads: statistics.median(values)
        for (size, candidate, threads), values in data.items()
        if size == n and candidate == version
    }


def make_plots(perf, cachegrind, output_dir):
    successful = {key: values for key, values in perf.items() if values}
    if not successful:
        raise ValueError("Nessuna misura perf valida (exit_status=0 e wall_seconds disponibile).")

    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted({n for n, _, _ in successful})
    for n in sizes:
        keys = [key for key in successful if key[0] == n]
        threads = sorted({thread_count for _, _, thread_count in keys})
        versions = ordered_versions(keys)
        wall_series = []
        wall_ranges = {}
        for version in versions:
            wall_series.append((version, median_by_thread(successful, n, version)))
            wall_ranges[version] = {
                thread_count: (min(values), max(values))
                for (size, candidate, thread_count), values in successful.items()
                if size == n and candidate == version
            }
        wall_max = max(high for ranges in wall_ranges.values() for _, high in ranges.values()) * 1.1
        svg_plot(
            output_dir / f"lu_wall_time_n{n}.svg",
            f"LU: tempo mediano per numero di thread (N={n})",
            "Tempo wall (s)",
            threads,
            wall_series,
            wall_max,
            error_ranges=wall_ranges,
        )

        baseline = median_by_thread(successful, n, "v0")
        speedup_series = []
        for version in versions:
            version_times = median_by_thread(successful, n, version)
            speedups = {
                thread_count: baseline[thread_count] / seconds
                for thread_count, seconds in version_times.items()
                if thread_count in baseline
            }
            speedup_series.append((version, speedups))
        speedup_values = [
            value for _, values in speedup_series for value in values.values()
        ]
        if baseline:
            svg_plot(
                output_dir / f"lu_speedup_n{n}.svg",
                f"LU: speedup rispetto a v0 (N={n})",
                "Speedup (v0 / versione)",
                threads,
                speedup_series,
                max(speedup_values, default=1) * 1.1,
            )
        else:
            print(f"Avviso: v0 mancante per N={n}; grafico speedup non generato.", file=sys.stderr)

        cg_values = {
            key: values
            for key, values in cachegrind.items()
            if key[0] == n and values
        }
        if cg_values:
            cg_versions = ordered_versions(cg_values)
            cg_series = [
                (version, median_by_thread(cg_values, n, version))
                for version in cg_versions
            ]
            cg_max = max(value for _, values in cg_series for value in values.values()) * 1.1
            scale = 1_000_000 if cg_max >= 1_000_000 else 1
            suffix = " M" if scale > 1 else ""
            svg_plot(
                output_dir / f"lu_cachegrind_instructions_n{n}.svg",
                f"LU: istruzioni Cachegrind (N={n})",
                "Istruzioni simulate",
                sorted({thread_count for _, _, thread_count in cg_values}),
                cg_series,
                cg_max,
                scale=scale,
                y_suffix=suffix,
            )

    if len(sizes) > 1:
        all_threads = sorted({threads for _, _, threads in successful})
        versions = ordered_versions(successful)
        for thread_count in all_threads:
            wall_series = []
            wall_ranges = {}
            speedup_series = []
            for version in versions:
                medians = {
                    n: statistics.median(values)
                    for (n, candidate, threads), values in successful.items()
                    if candidate == version and threads == thread_count
                }
                if not medians:
                    continue
                wall_series.append((version, medians))
                wall_ranges[version] = {
                    n: (min(successful[(n, version, thread_count)]),
                        max(successful[(n, version, thread_count)]))
                    for n in medians
                }
                speedups = {
                    n: statistics.median(successful[(n, "v0", thread_count)]) / median
                    for n, median in medians.items()
                    if (n, "v0", thread_count) in successful
                }
                if speedups:
                    speedup_series.append((version, speedups))

            bounds = [bounds for ranges in wall_ranges.values() for bounds in ranges.values()]
            svg_plot(
                output_dir / f"lu_wall_time_vs_n_t{thread_count}.svg",
                f"LU: tempo mediano per dimensione ({thread_count} thread)",
                "Tempo wall (s, scala logaritmica)",
                sizes,
                wall_series,
                max(high for _, high in bounds),
                error_ranges=wall_ranges,
                x_label="Dimensione N (categorie equidistanti)",
                log_y_min=min(low for low, _ in bounds),
            )
            if speedup_series:
                speedup_max = max(
                    value for _, values in speedup_series for value in values.values()
                )
                svg_plot(
                    output_dir / f"lu_speedup_vs_n_t{thread_count}.svg",
                    f"LU: speedup per dimensione ({thread_count} thread)",
                    "Speedup (v0 / versione)",
                    sizes,
                    speedup_series,
                    speedup_max * 1.1,
                    x_label="Dimensione N (categorie equidistanti)",
                )

    return sizes


def main():
    parser = argparse.ArgumentParser(
        description="Genera grafici SVG dai risultati CSV di perf_valgrind.sh."
    )
    parser.add_argument("csv_file", nargs="?", type=Path, default=Path("lu_results.csv"))
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="cartella dei grafici (default: <nome_csv>_plots accanto al CSV)",
    )
    args = parser.parse_args()
    csv_path = args.csv_file
    output_dir = args.output_dir or csv_path.with_name(f"{csv_path.stem}_plots")

    try:
        perf, cachegrind = parse_csv(csv_path)
        sizes = make_plots(perf, cachegrind, output_dir)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))

    print(f"Grafici per N={', '.join(map(str, sizes))} salvati in: {output_dir}")
    print("Usare il grafico wall-time per i tempi e quello speedup per il confronto con v0.")


if __name__ == "__main__":
    main()
