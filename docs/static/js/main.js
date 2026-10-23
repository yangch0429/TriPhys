// TriPhysBench project page: leaderboard and small interactions.
(function () {
  const ICON = { gemini: 'icon_gemini.png', openai: 'icon_openai.png', qwen: 'icon_qwen.png', sensenova: 'icon_sensenova.png' };

  // ---------------- data (paper Table III and Table V, scores x100) ----------------
  const SUB = (s) => s.replace(/S_(\w+)/g, 'S<sub>$1</sub>');
  const T3 = {
    groups: [['MPR', 5, 'g1'], ['KPR', 5, 'g2'], ['Overall', 3, '']],
    cols: ['ACC', 'S_obs', 'S_formula', 'S_derive', 'OGS', 'ACC', 'S_obs', 'S_formula', 'S_derive', 'OGS', 'ACC', 'SCS', 'OGS'],
    sections: [
      ['Closed-source models', [
        ['GPT-5.5', 'openai', [43.16, 42.00, 92.87, 61.62, 49.02, 45.97, 53.98, 91.35, 71.43, 53.41, 44.41, 66.27, 50.97]],
        ['Gemini-3.6-Flash', 'gemini', [42.67, 48.68, 90.02, 63.27, 49.04, 60.04, 59.34, 92.58, 74.86, 64.39, 50.38, 68.62, 55.85]],
      ]],
      ['Open-weight models', [
        ['InteractiveOmni-4B', 'sensenova', [36.50, 35.31, 81.69, 49.56, 40.97, 27.70, 42.72, 71.15, 51.65, 34.96, 32.59, 51.63, 38.30]],
        ['Qwen3.5-4B', 'qwen', [37.39, 38.49, 82.46, 50.55, 42.14, 41.51, 58.52, 88.46, 66.48, 49.25, 39.22, 59.47, 45.30]],
        ['Qwen3.5-9B', 'qwen', [39.24, 40.35, 87.50, 54.50, 44.59, 45.36, 58.93, 89.70, 69.37, 52.54, 41.96, 62.49, 48.12]],
        ['Qwen3.6-27B', 'qwen', [41.44, 43.53, 89.04, 55.70, 46.57, 46.72, 58.52, 92.03, 70.05, 53.73, 43.78, 63.66, 49.75]],
      ]],
    ],
    note: 'Table III of the paper: zero-shot results on all 410 questions in the native tool loop with web search and Python. ' +
      'ACC: answer correctness with partial credit. S_obs / S_formula / S_derive: solution-score components for variable identification, relation selection and numerical derivation. ' +
      'SCS is judged by DeepSeek-V4-Flash; OGS = 0.7 ACC + 0.3 SCS. Bold marks the best score in each column. All scores ×100.',
  };
  const T5 = {
    groups: [['MPR', 1, 'g1'], ['KPR', 1, 'g2'], ['Overall', 3, '']],
    cols: ['ACC', 'ACC', 'ACC', 'SCS', 'OGS'],
    sections: [
      ['Closed-source models', [
        ['GPT-5.5', 'openai', [51.71, 50.43, 51.14, 69.07, 56.52]],
        ['Gemini-3.6-Flash', 'gemini', [53.28, 66.34, 59.10, 70.42, 62.50]],
      ]],
      ['Open-weight models', [
        ['Qwen3.5-4B', 'qwen', [44.12, 49.32, 46.44, 62.51, 51.26], 'base'],
        ['TriPhys-4B', 'ours', [51.79, 57.29, 54.24, 68.28, 58.45], [17.4, 16.2, 16.8, 9.2, 14.0]],
        ['Qwen3.5-9B', 'qwen', [41.17, 54.05, 46.90, 66.18, 52.68], 'base'],
        ['TriPhys-9B', 'ours', [48.87, 61.69, 54.58, 70.16, 59.25], [18.7, 14.1, 16.4, 6.0, 12.5]],
        ['Qwen3.6-27B', 'qwen', [44.71, 55.25, 49.41, 67.84, 54.94], 'base'],
        ['TriPhys-27B', 'ours', [51.54, 66.46, 58.18, 71.56, 62.19], [15.3, 20.3, 17.7, 5.5, 13.2]],
      ]],
    ],
    note: 'Table V of the paper: 110 questions from 53 videos excluded from training. Green rows are TriPhys models, each listed under its backbone; ' +
      'the green number below each score is the relative gain (%) over that backbone. Bold marks the best score in each column. All scores ×100.',
  };
  const BOARDS = { t3: T3, t5: T5 };

  function renderBoard(key) {
    const B = BOARDS[key];
    const table = document.getElementById('board');
    table.classList.toggle('dense', B.cols.length > 8);
    const all = B.sections.flatMap((s) => s[1]);
    const max = B.cols.map((_, j) => Math.max(...all.map((r) => r[2][j])));
    let j0 = 0;
    const colClass = [];
    B.groups.forEach(([, n, g]) => { for (let k = 0; k < n; k++) colClass.push((g ? 'grp ' + g : '') + (k === 0 ? ' gstart' : '')); });
    table.querySelector('thead').innerHTML =
      '<tr class="ghead"><th rowspan="2" style="text-align:left">Model</th>' +
      B.groups.map(([name, n, g]) => `<th colspan="${n}" class="gstart ${g}">${name}</th>`).join('') + '</tr>' +
      '<tr>' + B.cols.map((c, j) => `<th class="${colClass[j]}">${SUB(c)}</th>`).join('') + '</tr>';
    const ncol = B.cols.length + 1;
    table.querySelector('tbody').innerHTML = B.sections.map(([title, rows]) =>
      `<tr class="sec"><td colspan="${ncol}">${title}</td></tr>` + rows.map((r) => {
        const ours = r[1] === 'ours';
        const icon = ours ? '<span class="badge">OURS</span>' : `<img src="static/images/${ICON[r[1]]}" alt="">`;
        const cells = r[2].map((v, j) => {
          const best = Math.abs(v - max[j]) < 1e-9;
          const gain = ours ? `<span class="gain">+${r[3][j].toFixed(1)}</span>` : '';
          return `<td class="${colClass[j]}"><span class="cell${best ? ' best' : ''}">${v.toFixed(2)}${gain}</span></td>`;
        }).join('');
        return `<tr class="${ours ? 'ours' : r[3] === 'base' ? 'base' : ''}"><td><span class="model${ours ? ' indent' : ''}">${icon}${r[0]}</span></td>${cells}</tr>`;
      }).join('')).join('');
    document.getElementById('board-note').innerHTML = SUB(B.note);
  }

  let current = /[?&]board=t5/.test(location.search) ? 't5' : 't3';
  document.querySelectorAll('.tabs button').forEach((x) => x.classList.toggle('on', x.dataset.board === current));
  document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.tabs button').forEach((x) => x.classList.remove('on'));
    b.classList.add('on');
    current = b.dataset.board;
    renderBoard(current);
  }));
  renderBoard(current);
  // ---------------- nav state ----------------
  const nav = document.getElementById('nav');
  const links = [...document.querySelectorAll('.nav-links a')];
  const secs = links.map((a) => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  function onScroll() {
    nav.classList.toggle('scrolled', window.scrollY > 10);
    let cur = null;
    secs.forEach((sec) => { if (sec.getBoundingClientRect().top < 120) cur = sec.id; });
    links.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + cur));
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // ---------------- reveal on scroll ----------------
  if ('IntersectionObserver' in window && !/[?&]static/.test(location.search)) {
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    }), { threshold: 0.08 });
    document.querySelectorAll('.reveal').forEach((el) => io.observe(el));
  } else {
    document.querySelectorAll('.reveal').forEach((el) => el.classList.add('in'));
  }

  // ---------------- lightbox ----------------
  const lb = document.getElementById('lightbox');
  document.querySelectorAll('.figure img, details img').forEach((img) => img.addEventListener('click', () => {
    lb.querySelector('img').src = img.src;
    lb.classList.add('on');
  }));
  lb.addEventListener('click', () => lb.classList.remove('on'));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') lb.classList.remove('on'); });

  // ---------------- copy buttons ----------------
  document.querySelectorAll('.copy').forEach((b) => b.addEventListener('click', () => {
    const text = document.getElementById(b.dataset.copy).innerText;
    navigator.clipboard.writeText(text).then(() => {
      b.textContent = 'Copied!';
      setTimeout(() => (b.textContent = 'Copy'), 1500);
    });
  }));
})();
