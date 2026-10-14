/*
 * 大屏"视频文件播放联动"回归自检（无需浏览器）
 *
 * 用法：先起大屏，再跑本脚本
 *   uv run python dashboard.py --config config.yaml --port 8023
 *   node tools/frontend_probe.js [http://127.0.0.1:8023]
 *
 * 做法：从运行中的大屏拉取首页 HTML，抽出内嵌的真实脚本，在最小 DOM 沙箱里执行，
 * 然后用真实 HTTP 请求驱动 onMarkerVideoTime(t)，打印每个时刻的窗口范围、曲线点数
 * 与分析表数值。目的是验证"播放时间 → 时移曲线 / 振动分析"确实联动且口径自洽。
 *
 * 注意：这里是 DOM stub，不是浏览器。tick 里的 renderLiveMedia 在真实浏览器中
 * 有"同相机同源不重建"的提前返回，stub 下不存在该状态，故开头先等 tick 落定
 * 再开始驱动，以免它的异步尾巴与本次驱动交错。
 */
const vm = require('vm');

const BASE = (process.argv[2] || 'http://127.0.0.1:8023').replace(/\/$/, '');

// ---------------- 最小 DOM stub ----------------
const els = {};
// 每个 canvas 一份"绘制记录"：clearRect 清空、fillText 累积。用于检查频谱图在
// 未满窗时究竟画了曲线还是画了"等待首窗数据…"（曲线会走 lineTo，不产生 fillText）。
const drawRec = {};
function ctxOf(id) {
  if (!drawRec[id]) drawRec[id] = { fills: [] };
  const store = drawRec[id];
  return new Proxy({}, {
    get: (t, k) => {
      if (k === 'measureText') return () => ({ width: 10 });
      if (k === 'canvas') return { width: 300, height: 120 };
      if (k === 'clearRect') return () => { store.fills = []; };
      if (k === 'fillText') return (txt) => { store.fills.push(String(txt)); };
      return () => {};
    },
    set: () => true,
  });
}
function mkEl(id) {
  const e = {
    id, dataset: {}, style: {}, _html: '', textContent: '', value: '',
    _ev: {},   // 真实记录监听器，供 fire() 模拟 DOM 事件（timeupdate / pause / ended）
    clientWidth: 300, clientHeight: 120, width: 300, height: 120,
    classList: { contains: () => true, toggle() {}, add() {}, remove() {} },
    querySelector: (s) => getEl('q:' + id + s),
    querySelectorAll: () => [],
    addEventListener(ev, fn) { (e._ev[ev] = e._ev[ev] || []).push(fn); },
    appendChild() {}, removeChild() {},
    getAttribute() { return null; }, setAttribute() {}, removeAttribute() {},
    getContext: () => ctxOf(id),
  };
  Object.defineProperty(e, 'innerHTML', { get: () => e._html, set: (v) => { e._html = v; } });
  return e;
}
function getEl(id) { if (!els[id]) els[id] = mkEl(id); return els[id]; }

function cellsOf(htmlStr) {
  // 表格行含换行，用 [\s\S]；每 5 列成一行（目标/振幅/主频/挠度/索力）
  // td 可能带属性（如占位列的 class="ph"），用 <td[^>]*> 而不是 <td>
  const tds = [...htmlStr.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/g)]
    .map(m => m[1].replace(/<[^>]+>/g, '').trim());
  const rows = [];
  for (let i = 0; i < tds.length; i += 5) rows.push(tds.slice(i, i + 5));
  return rows;
}

(async () => {
  const page = await (await fetch(BASE + '/')).text();
  const m = [...page.matchAll(/<script>([\s\S]*?)<\/script>/g)];
  if (!m.length) throw new Error('页面里没找到内嵌脚本');
  const code = m[0][1];

  const sandbox = {
    console,
    drawRec,   // 绘制记录（在 Node 侧持有），供 __api.snap 读取频谱图实际画了什么
    document: {
      getElementById: getEl,
      querySelector: (s) => getEl('q:' + s),
      querySelectorAll: () => [],
      createElement: (t) => mkEl('new:' + t),
    },
    window: {},
    // Node 的 fetch 不吃相对 URL（浏览器里没问题），补成绝对地址
    fetch: (u, o) => fetch(typeof u === 'string' && u.startsWith('/') ? BASE + u : u, o),
    setInterval: () => 0, setTimeout: () => 0, clearInterval() {},
    Date, Math, JSON, Number, String, Object, Array, isFinite, parseInt, parseFloat,
  };
  vm.createContext(sandbox);
  vm.runInContext(code + `
;globalThis.__api = {
  drive: (t, paused) => onMarkerVideoTime({ currentTime: t, paused: !!paused }, true),
  env: (c, isFile) => { markerLiveCam = c; markerIsFile = isFile; },
  // 模拟真实 DOM 事件（走 addEventListener 挂上的真实监听器，而不是直接调函数）
  listeners: () => Object.keys(document.getElementById('q:markerLiveMediavideo')._ev || {}),
  fire: (ev, props) => {
    const el = document.getElementById('q:markerLiveMediavideo');
    // 监听器是 () => onMarkerVideoTime(v) 形式、闭包捕获的就是这个元素，
    // 所以把 currentTime/paused/ended 写到元素上，它们才读得到。
    Object.assign(el, props);
    const fns = (el._ev && el._ev[ev]) || [];
    fns.forEach(f => f({ target: el }));
    return fns.length;
  },
  snap: () => ({
    n: (lastShiftList || []).map(s => ({ name: s.name, n: s.data.length })),
    win: (document.getElementById('markerWinInfo') || {}).textContent,
    ana: (document.querySelector('#markerAnaTbl tbody') || {}).innerHTML,
    ana_time: (document.getElementById('anaTime') || {}).textContent,
    syncing: markerSyncing, winT: markerWinT, err: markerWinErr,
    gate: { ready: chartGate.ready, reason: chartGate.reason,
            cover: chartGate.cover, need: chartGate.need },
    spec: (drawRec['chartSpectrum'] || { fills: [] }).fills.slice(),
  }),
};
`, sandbox);

  const api = sandbox.__api;
  // 页面脚本末尾会立即跑一次 tick()（异步），先等它落定再驱动
  await new Promise(r => setTimeout(r, 400));
  api.env('marker_cam_01', true);
  console.log(`探测目标：${BASE}\n`);

  // 覆盖"未满窗 → 刚满窗 → 末尾"三段，重点看跨过门限那一跳
  const MARK = [2, 5, 9.8, 10.5, 11.84];
  for (const t of MARK) {
    await api.drive(t, false);
    await new Promise(r => setTimeout(r, 60));
    const s = api.snap();
    console.log(`--- 播放到 t=${t}s ---`);
    console.log(`  窗口提示 : ${s.win || '(空)'}`);
    console.log(`  状态     : syncing=${s.syncing} markerWinT=${s.winT} 请求失败=${s.err}`);
    console.log(`  曲线点数 : ${s.n.map(x => x.name + '=' + x.n).join('  ')}`);
    console.log(`  频谱门限 : ready=${s.gate.ready} cover=${s.gate.cover} need=${s.gate.need}`
      + ` reason=${JSON.stringify(s.gate.reason)}`);
    console.log(`  频谱图   : ${s.spec.length ? JSON.stringify(s.spec) : '(什么都没画)'}`);
    console.log(`  分析时刻 : ${s.ana_time}`);
    cellsOf(s.ana).forEach(r => console.log(`  分析表   : ${r.join(' | ')}`));
    console.log('');
  }

  await api.drive(11.84, true);
  await new Promise(r => setTimeout(r, 60));
  const p = api.snap();
  console.log('--- 暂停态 ---');
  console.log(`  窗口提示 : ${p.win}`);

  const before = api.snap();
  await api.drive(11.84, true);
  const after = api.snap();
  console.log(`\n节流检查：重复驱动同一 t 结果一致 = ${before.ana === after.ana}`);

  // ---- 关键回归：模拟播放末尾（timeupdate 节流 + 临近末尾强制对齐） ----
  // 实测问题：视频播完后面板停在 M1 0.088Hz/SNR 2540.1、M2 0.128Hz/SNR 301.8，
  // 反查对应窗口终点 t=11.36s，而数据实际到 11.84s。原因是末尾几次 timeupdate
  // 的增量小于 0.4s 节流门限被一直跳过；又因 <video> 带 loop，回绕时不触发 ended，
  // 所以单靠 ended 兜不住。现在改为"离末尾不足 0.5s 时强制刷新"。
  console.log('\n--- 模拟播放末尾（走真实 addEventListener 挂上的监听器）---');
  console.log(`  视频元素已挂监听: ${JSON.stringify(api.listeners())}`);
  const DUR = 11.88;
  const seq = [
    ['timeupdate', 11.00, '距上次刷新 0.84s → 通过'],
    ['timeupdate', 11.36, '距上次仅 0.36s 且离末尾 0.52s → 应被节流跳过（复现点）'],
    ['timeupdate', 11.60, '离末尾 0.28s → 强制刷新'],
    ['timeupdate', 11.84, '离末尾 0.04s → 强制刷新（数据末端）'],
    ['loop', 0, 'loop 回绕：该事件无人监听，界面保持末端结果'],
  ];
  for (const [ev, t, why] of seq) {
    const n = api.fire(ev, { currentTime: t, duration: DUR, paused: false, ended: false });
    await new Promise(r => setTimeout(r, 200));
    const s = api.snap();
    console.log(`  ${ev.padEnd(10)} t=${String(t).padEnd(5)} (${n} 个监听器) → ${s.win.padEnd(22)} // ${why}`);
  }
  const fin = api.snap();
  console.log('  最终分析表：');
  cellsOf(fin.ana).forEach(r => console.log(`    ${r.join(' | ')}`));
  console.log('  ✅ 期望与 spectrum.jsonl 的 dy 行逐位一致：M1 0.0842 / SNR 3741.4，M2 0.1202 / 389.6');

  console.log('\n判定要点：t<10s 时频谱图应只出现“等待首窗数据…”且主频为 "—"（不出曲线）；'
    + 't>=10s 才出频谱，且图上主频与表格主频一致；'
    + '播放末尾（ended）必须强制刷新到数据末端，不得停在中途窗口；'
    + '最大振幅/主频随窗口变化（真实计算），挠度/索力恒为占位值（config.marker.placeholder）。');
  process.exit(0);
})().catch(e => { console.error('探针异常:', e); process.exit(1); });
