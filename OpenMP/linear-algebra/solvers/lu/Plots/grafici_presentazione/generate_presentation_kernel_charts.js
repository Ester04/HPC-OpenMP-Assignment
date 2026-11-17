#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const root = __dirname;
const output = path.join(root, 'grafici_presentazione');
const lines = fs.readFileSync(path.join(root, 'lu_kernel_results.csv'), 'utf8').trim().split(/\r?\n/);
const header = lines.shift().split(',');
const rows = lines.map((line, index) => {
  const cells = line.split(',');
  if (cells.length !== header.length) throw new Error(`Invalid CSV row ${index + 2}`);
  return Object.fromEntries(header.map((name, i) => [name, cells[i]]));
}).filter(row => row.tool === 'perf' && row.exit_status === '0' && row.threads === '4');

const sizes = [32, 128, 1024, 2000, 4000];
const versions = [
  ['v0', 'Original (serial)', '#B5A5FF'],
  ['v2', 'OpenMP', '#BEC4D9'],
  ['v2_simd', 'SIMD', '#FF647C'],
  ['v2_recip', '+ reciprocal', '#E6B3FF'],
  ['v2_hoist', '+ hoisting', '#CF9EF0'],
  ['v2_rows', '+ row pointers', '#BC8CDE'],
  ['v2_restrict', '+ restrict', '#A979CD'],
  ['v3', 'static', '#55D6E8'],
  ['v4', 'static,1', '#9DE3C3'],
];

function median(version, size, field) {
  const samples = rows.filter(r => r.version === version && Number(r.n) === size).map(r => {
    if (r[field] === '' || !Number.isFinite(Number(r[field])) || Number(r[field]) < 0) {
      throw new Error(`Invalid ${field}: ${version}, N=${size}`);
    }
    return Number(r[field]);
  }).sort((a, b) => a - b);
  if (samples.length !== 3) throw new Error(`Expected three ${version} samples at N=${size}`);
  return samples[1];
}

function canvas(title, subtitle, body, footer, desc) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720" fill="#F7F8FC" role="img" aria-labelledby="title desc">
<title id="title">${title}</title><desc id="desc">${desc}</desc>
<style>text { font-family: Arial, Helvetica, sans-serif; } .muted { fill: #BEC4D9; }</style>
<rect width="1200" height="720" rx="24" fill="#151832"/>
<text x="600" y="46" text-anchor="middle" font-size="29" font-weight="700">${title}</text>
<text x="600" y="79" text-anchor="middle" font-size="17" class="muted">${subtitle}</text>
${body}
<rect x="65" y="615" width="1070" height="82" rx="14" fill="#1C2040"/>
<text x="90" y="647" font-size="16" class="muted">${footer[0]}</text>
<text x="90" y="675" font-size="16" class="muted">${footer[1]}</text>
</svg>\n`;
}

function save(name, svg) {
  const file = path.join(output, name);
  if (fs.existsSync(file)) throw new Error(`Refusing to overwrite ${file}`);
  fs.writeFileSync(file, svg, { flag: 'wx' });
}

function times(field, title, filename, footer) {
  const origin = 260, unit = 11, max = 70;
  let body = '';
  for (let t = 0; t <= max; t += 10) {
    const x = origin + unit * t;
    body += `<line x1="${x}" y1="110" x2="${x}" y2="530" stroke="#3A4164" stroke-width="1.5"/><text x="${x}" y="558" text-anchor="middle" font-size="16" class="muted">${t}</text>\n`;
  }
  const baseline = median('v2_simd', 4000, field);
  versions.forEach(([version, label, color], i) => {
    if (field === 'kernel_seconds' && version === 'v2') return;
    const value = median(version, 4000, field);
    const y = 124 + i * 44, w = value * unit;
    const delta = 100 * (1 - value / baseline);
    const shown = `${delta >= 0 ? '+' : ''}${delta.toFixed(2)}%`;
    body += `<text x="244" y="${y + 17}" text-anchor="end" font-size="16" class="muted">${label}</text><rect x="${origin}" y="${y}" width="${w.toFixed(4)}" height="23" rx="7" fill="${color}"/><text x="${(origin + w + 9).toFixed(2)}" y="${y + 17}" font-size="16" font-weight="700">${value.toFixed(field === 'kernel_seconds' ? 6 : 3)} s</text><text x="1080" y="${y + 17}" text-anchor="end" font-size="15" class="muted">${shown}</text>\n`;
  });
  body += '<text x="1080" y="110" text-anchor="end" font-size="14" class="muted">Time saved vs SIMD</text><text x="645" y="590" text-anchor="middle" font-size="17" class="muted">Seconds · lower is better</text>';
  save(filename, canvas(title, 'N = 4000 · 4 threads requested · median of 3 perf runs',
    body, [footer, field === 'kernel_seconds'
      ? 'Time saved = (1 − median time / SIMD median) × 100%; positive = faster, negative = slower.'
      : 'Time saved = (1 − median time / median SIMD time) × 100%; original is serial.'],
    `Median ${field} by version. Percentages show time reduction relative to SIMD.`));
}

times('kernel_seconds', 'LU kernel time across optimization stages',
  'kernel_time_optimizations_n4000.svg',
  'Kernel time includes OpenMP team creation and synchronization, but excludes initialization.');
times('wall_seconds', 'Whole-process wall time across optimization stages',
  'wall_time_optimizations_n4000.svg',
  'Wall time covers perf and the whole program; timer resolution and overhead may affect it.');

let deltaBody = '';
const x0 = 155, x1 = 1060, zeroY = 462, scale = 6;
for (let t = 0; t <= 50; t += 10) {
  const y = zeroY - t * scale;
  deltaBody += `<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="#3A4164" stroke-width="${t === 0 ? 2 : 1.4}"/><text x="140" y="${y + 5}" text-anchor="end" font-size="16" class="muted">${t}%</text>`;
}
sizes.forEach((n, i) => {
  const a = median('v3', n, 'kernel_seconds'), b = median('v4', n, 'kernel_seconds');
  const d = 100 * (b / a - 1), x = 235 + 185 * i, y = zeroY - d * scale;
  deltaBody += `<rect x="${x - 30}" y="${Math.min(y, zeroY).toFixed(3)}" width="60" height="${Math.max(1, Math.abs(d * scale)).toFixed(3)}" fill="${d < 0 ? '#9DE3C3' : '#FF647C'}"/><text x="${x}" y="${(d >= 0 ? y - 10 : zeroY + 22).toFixed(3)}" text-anchor="middle" font-size="18" font-weight="700">${d >= 0 ? '+' : ''}${d.toFixed(2)}%</text><text x="${x}" y="523" text-anchor="middle" font-size="17" class="muted">${n}</text><text x="${x}" y="551" text-anchor="middle" font-size="13" class="muted">${a.toFixed(6)} / ${b.toFixed(6)} s</text>`;
});
deltaBody += '<text x="600" y="586" text-anchor="middle" font-size="16" class="muted">Matrix size N · labels: static / static,1 median kernel time</text>';
save('static_vs_static1_kernel_by_dataset.svg', canvas('static vs. static,1 by dataset',
  'Kernel-time change: (median static,1 / median static − 1) × 100%',
  deltaBody, ['Negative values mean static,1 is faster; 4 threads requested, 3 perf runs.',
    'Small datasets are sensitive to timer resolution and OpenMP startup overhead.'],
  'Percentage change in median kernel time between static,1 and static for each dataset.'));

let cacheBody = '';
const cacheOrigin = 190, cacheScale = 3.2;
for (let t = 0; t <= 250; t += 50) {
  const x = cacheOrigin + t * cacheScale;
  cacheBody += `<line x1="${x}" y1="130" x2="${x}" y2="515" stroke="#3A4164" stroke-width="1.5"/><text x="${x}" y="540" text-anchor="middle" font-size="15" class="muted">${t}%</text>`;
}
sizes.forEach((n, i) => {
  const a = median('v3', n, 'cache_misses'), b = median('v4', n, 'cache_misses');
  const ratio = 100 * b / a, y = 140 + i * 74;
  cacheBody += `<text x="175" y="${y + 26}" text-anchor="end" font-size="18" class="muted">N=${n}</text><rect x="${cacheOrigin}" y="${y}" width="${(100 * cacheScale).toFixed(2)}" height="22" rx="5" fill="#55D6E8"/><rect x="${cacheOrigin}" y="${y + 27}" width="${(ratio * cacheScale).toFixed(2)}" height="22" rx="5" fill="#9DE3C3"/><text x="1095" y="${y + 17}" text-anchor="end" font-size="15" fill="#55D6E8">${a.toLocaleString('en-US')} misses</text><text x="1095" y="${y + 44}" text-anchor="end" font-size="15" fill="#9DE3C3">${b.toLocaleString('en-US')} misses</text>`;
});
cacheBody += '<rect x="395" y="98" width="18" height="18" rx="4" fill="#55D6E8"/><text x="423" y="113" font-size="17">static</text><rect x="590" y="98" width="18" height="18" rx="4" fill="#9DE3C3"/><text x="618" y="113" font-size="17">static,1</text>';
cacheBody += '<text x="590" y="576" text-anchor="middle" font-size="16" class="muted">Median cache misses relative to static for the same N (static = 100%)</text>';
save('cache_misses_static_vs_static1_by_dataset.svg', canvas('Cache misses: static vs. static,1',
  'Median perf counts · 4 threads requested · 3 runs · static / static,1',
  cacheBody, ['Values on the right are absolute median counts.',
    'Counts cover the whole process and runtime; hardware event semantics depend on the CPU.'],
  'Cache miss medians by dataset, normalized per dataset to static equals 100 percent; absolute counts are labeled.'));
console.log(`Created four SVG charts in ${output}`);
