// Interactive ROI rectangle (draw / move / resize) over a scaled frame image.
// ROIs are stored in FULL-resolution video coordinates and converted to the
// displayed pixel size on the fly, so they stay correct regardless of zoom.
(function () {
  const cams = {};                       // video_id -> {x,y,w,h} in full-res coords

  function setStatus(msg) { document.getElementById('status').textContent = msg || ''; }

  function initCam(card) {
    const vid = card.dataset.video;
    const fullW = +card.dataset.fullw, fullH = +card.dataset.fullh;
    const stage = card.querySelector('.roi-stage');
    const img = card.querySelector('.roi-frame');
    const box = card.querySelector('.roi-box');
    const readout = card.querySelector('.roi-readout');

    const sx = () => img.clientWidth / fullW;   // display px per full px (x)
    const sy = () => img.clientHeight / fullH;

    function render() {
      const r = cams[vid];
      if (!r) { box.hidden = true; return; }
      box.hidden = false;
      box.style.left = (r.x * sx()) + 'px';
      box.style.top = (r.y * sy()) + 'px';
      box.style.width = (r.w * sx()) + 'px';
      box.style.height = (r.h * sy()) + 'px';
      readout.textContent = `x${r.x} y${r.y} w${r.w} h${r.h}`;
    }
    function setFull(r) {
      r.x = Math.max(0, Math.min(r.x, fullW - 2));
      r.y = Math.max(0, Math.min(r.y, fullH - 2));
      r.w = Math.max(2, Math.min(r.w, fullW - r.x));
      r.h = Math.max(2, Math.min(r.h, fullH - r.y));
      cams[vid] = r; render();
    }
    card._setFull = setFull;
    card._render = render;

    if (card.dataset.roi) {
      const [x, y, w, h] = card.dataset.roi.split(',').map(Number);
      cams[vid] = { x, y, w, h };
    }
    img.addEventListener('load', render);
    if (img.complete) render();
    window.addEventListener('resize', render);

    // pointer interactions ------------------------------------------------- //
    let mode = null, startPx = null, startRoi = null;
    function pos(e) {
      const b = img.getBoundingClientRect();
      return { x: e.clientX - b.left, y: e.clientY - b.top };
    }
    stage.addEventListener('pointerdown', (e) => {
      const handle = e.target.closest('.h');
      const onBox = e.target === box || e.target.closest('.roi-box');
      startPx = pos(e);
      startRoi = cams[vid] ? { ...cams[vid] } : null;
      if (handle) { mode = 'resize-' + handle.className.split(' ')[1]; }
      else if (onBox && startRoi) { mode = 'move'; }
      else {                                    // draw new
        mode = 'draw';
        setFull({ x: startPx.x / sx(), y: startPx.y / sy(), w: 2, h: 2 });
        startRoi = { ...cams[vid] };
      }
      stage.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    stage.addEventListener('pointermove', (e) => {
      if (!mode) return;
      const p = pos(e);
      const dxF = (p.x - startPx.x) / sx(), dyF = (p.y - startPx.y) / sy();
      let r = { ...startRoi };
      if (mode === 'move') { r.x += dxF; r.y += dyF; }
      else if (mode === 'draw' || mode === 'resize-br') { r.w = startRoi.w + dxF; r.h = startRoi.h + dyF; }
      else if (mode === 'resize-tl') { r.x += dxF; r.y += dyF; r.w -= dxF; r.h -= dyF; }
      else if (mode === 'resize-tr') { r.y += dyF; r.w += dxF; r.h -= dyF; }
      else if (mode === 'resize-bl') { r.x += dxF; r.w -= dxF; r.h += dyF; }
      if (mode === 'draw') { r.x = startRoi.x; r.y = startRoi.y; }   // anchor draw origin
      if (r.w < 2) r.w = 2; if (r.h < 2) r.h = 2;
      setFull(r);
    });
    const end = (e) => { mode = null; };
    stage.addEventListener('pointerup', end);
    stage.addEventListener('pointercancel', end);

    // auto-locate ---------------------------------------------------------- //
    card.querySelector('.auto-btn').addEventListener('click', async () => {
      setStatus(`${vid}: locating light…`);
      try {
        const res = await fetch(card.dataset.autoroi, { method: 'POST' });
        const data = await res.json();
        if (data.ok) { setFull(data.roi); setStatus(`${vid}: located.`); }
        else { setStatus(`${vid}: ${data.message}`); }
      } catch (err) { setStatus(`${vid}: ${err}`); }
    });
  }

  document.querySelectorAll('.roi-cam').forEach(initCam);

  document.getElementById('save-btn').addEventListener('click', async () => {
    const payload = {};
    document.querySelectorAll('.roi-cam').forEach(card => {
      const r = cams[card.dataset.video];
      if (r) payload[card.dataset.video] = {
        x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.w), h: Math.round(r.h) };
    });
    if (Object.keys(payload).length === 0) { setStatus('Draw at least one ROI first.'); return; }
    setStatus('Saving…');
    const res = await fetch(window.SAVE_URL, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload) });
    if (res.ok) window.location = window.DETECT_URL;
    else setStatus('Save failed.');
  });
})();
