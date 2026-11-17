#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const root = __dirname;
const output = path.join(root, 'grafici_relazione_kernel');
const lines = fs.readFileSync(path.join(root, 'lu_kernel_results.csv'), 'utf8').trim().split(/\r?\n/);
const header = lines.shift().split(',');
const rows = lines.map((line, index) => {
  const cells = line.split(',');
  if (cells.length !== header.length) throw new Error(`Riga CSV ${index + 2} non valida`);
  return Object.fromEntries(header.map((name, i) => [name, cells[i]]));
}).filter(row => row.tool === 'perf' && row.exit_status === '0' && row.threads === '4');

function median(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  if (sorted.length !== 3 || sorted.some(value => !Number.isFinite(value))) {
    throw new Error('Sono richieste esattamente tre misure numeriche per configurazione');
  }
  return sorted[1];
}
function values(version, n, field) {
  return rows.filter(r => r.version === version && Number(r.n) === n).map(r => {
    if (field === 'ipc') {
      if (!r.instructions || !r.cycles || Number(r.cycles) <= 0) throw new Error('Contatori IPC mancanti');
      return Number(r.instructions) / Number(r.cycles);
    }
    if (!r[field]) throw new Error(`Campo ${field} mancante`);
    return Number(r[field]);
  });
}
const time = (version, n) => median(values(version, n, 'kernel_seconds'));
const colors = ['#64748b', '#a855f7', '#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e', '#0284c7', '#0d9488'];
const versions = [
  ['v0', 'Originale'], ['v2', 'OpenMP'], ['v2_simd', 'SIMD'],
  ['v2_recip', '+ reciproco'], ['v2_hoist', '+ hoisting'],
  ['v2_rows', '+ puntatori'], ['v2_restrict', '+ restrict'],
  ['v3', 'static'], ['v4', 'static,1'],
];
function wrap(title, subtitle, body, foot) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="600" viewBox="0 0 1000 600" role="img">
<title>${title}</title>
<rect width="1000" height="600" fill="#ffffff"/>
<g font-family="Arial, Helvetica, sans-serif" fill="#172033">
<text x="500" y="38" text-anchor="middle" font-size="25" font-weight="bold">${title}</text>
<text x="500" y="65" text-anchor="middle" font-size="15" fill="#475569">${subtitle}</text>
${body}
<text x="500" y="576" text-anchor="middle" font-size="13" fill="#475569">${foot}</text>
</g></svg>\n`;
}
function save(name, content) { fs.writeFileSync(path.join(output, name), content); }
function barChart(name, title, subtitle, data, max, unit, foot) {
  const x = 205, width = 670, top = 105, step = 46;
  let body = '';
  for (let i = 0; i <= 5; i++) {
    const px = x + width * i / 5;
    body += `<line x1="${px}" y1="90" x2="${px}" y2="525" stroke="#e2e8f0"/><text x="${px}" y="548" text-anchor="middle" font-size="13">${(max * i / 5).toFixed(max <= 2 ? 1 : 0)}</text>`;
  }
  data.forEach((d, i) => {
    const y = top + i * step, w = width * d.value / max;
    body += `<text x="${x - 12}" y="${y + 17}" text-anchor="end" font-size="16">${d.label}</text><rect x="${x}" y="${y}" width="${w.toFixed(2)}" height="23" fill="${d.color}"/><text x="${(x + w + 8).toFixed(2)}" y="${y + 17}" font-size="15">${d.value.toFixed(d.digits)}${unit}</text>`;
  });
  save(name, wrap(title, subtitle, body, foot));
}
fs.mkdirSync(output, { recursive: true });
barChart('tempi_passaggi_n4000.svg', 'Tempo del kernel per versione — N=4000',
  'Mediana di 3 esecuzioni perf · 4 thread richiesti · minore è meglio',
  versions.map(([version, label], i) => ({ label, value: time(version, 4000), digits: 3, color: colors[i] })),
  70, ' s', 'Originale seriale; le versioni successive sono cumulative.');
barChart('ipc_passaggi_n4000.svg', 'IPC per versione — N=4000',
  'Mediana dei rapporti istruzioni / cicli per esecuzione · 4 thread richiesti',
  versions.map(([version, label], i) => ({ label, value: median(values(version, 4000, 'ipc')), digits: 3, color: colors[i] })),
  2, '', 'Contatori dell’intero processo, non del solo kernel. IPC non è speedup.');

const sizes = [32, 128, 1024, 2000, 4000];
const chosen = [['v2', 'OpenMP', '#a855f7'], ['v2_simd', 'SIMD', '#ef4444'], ['v3', 'static', '#0284c7'], ['v4', 'static,1', '#0d9488']];
let speedBody = '';
const xAt = i => 150 + 175 * i;
const yAt = v => 480 - v * 160;
for (let i = 0; i <= 4; i++) {
  const y = yAt(i * 0.5);
  speedBody += `<line x1="120" y1="${y}" x2="890" y2="${y}" stroke="#e2e8f0"/><text x="110" y="${y + 5}" text-anchor="end" font-size="14">${(i * 0.5).toFixed(1)}×</text>`;
}
sizes.forEach((n, i) => { speedBody += `<text x="${xAt(i)}" y="508" text-anchor="middle" font-size="15">${n}</text>`; });
chosen.forEach(([v, label, color], j) => {
  const coords = sizes.map((n, i) => [xAt(i), yAt(time('v0', n) / time(v, n))]);
  speedBody += `<polyline points="${coords.map(p => p.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="3"/>`;
  coords.forEach(([x, y]) => { speedBody += `<circle cx="${x}" cy="${y.toFixed(2)}" r="5" fill="${color}"/>`; });
  speedBody += `<rect x="${160 + j * 175}" y="530" width="16" height="10" fill="${color}"/><text x="${182 + j * 175}" y="540" font-size="14">${label}</text>`;
});
save('speedup_per_dataset.svg', wrap('Speedup del kernel rispetto all’originale',
  'Rapporto tra le mediane · 4 thread richiesti · 1× = tempo uguale all’originale', speedBody,
  'Dataset equidistanti; originale seriale. Tre prove non stabiliscono significatività.'));

let deltaBody = '';
const zeroY = 455, scale = 6.5;
for (const tick of [0, 10, 20, 30, 40, 50]) {
  const y = zeroY - tick * scale;
  deltaBody += `<line x1="125" y1="${y}" x2="895" y2="${y}" stroke="#e2e8f0"/><text x="112" y="${y + 5}" text-anchor="end" font-size="14">${tick}%</text>`;
}
sizes.forEach((n, i) => {
  const delta = 100 * (time('v4', n) / time('v3', n) - 1), x = xAt(i), y = zeroY - delta * scale;
  deltaBody += `<rect x="${x - 30}" y="${Math.min(y, zeroY).toFixed(2)}" width="60" height="${Math.max(1, Math.abs(delta * scale)).toFixed(2)}" fill="${delta < 0 ? '#0d9488' : '#f97316'}"/><text x="${x}" y="${(delta >= 0 ? y - 10 : zeroY + 22).toFixed(2)}" text-anchor="middle" font-size="15">${delta >= 0 ? '+' : ''}${delta.toFixed(2)}%</text><text x="${x}" y="510" text-anchor="middle" font-size="15">${n}</text>`;
});
save('static_vs_static1_dataset.svg', wrap('Differenza di tempo: static,1 rispetto a static',
  '(mediana static,1 / mediana static − 1) × 100 · 4 thread richiesti', deltaBody,
  'Valori negativi: static,1 più veloce. N piccoli: timer e overhead possono incidere.'));
console.log(`Grafici generati in ${output}`);
