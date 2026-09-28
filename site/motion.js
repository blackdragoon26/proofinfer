/* proofinfer — a slow drift of dots behind the page.
 *
 * Deliberately quiet: low contrast, slow, and it never sits under text at a
 * density that competes with reading. Three things it will not do:
 *
 *   - move, if the reader has asked for reduced motion (prefers-reduced-motion)
 *   - burn CPU while the tab is in the background
 *   - touch the DOM; it is one canvas and nothing else
 */

(function () {
  "use strict";

  var canvas = document.getElementById("bg");
  if (!canvas || !canvas.getContext) return;

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  var ctx = canvas.getContext("2d");
  var dots = [];
  var w = 0;
  var h = 0;
  var raf = null;

  // Tuned by eye against the page: fewer, slower, fainter than looks right
  // in isolation.
  var COUNT = 46;        // dots on a large viewport
  var LINK = 118;        // px within which two dots are joined by a line
  var SPEED = 0.055;     // px per frame
  var ALPHA_DOT = 0.16;
  var ALPHA_LINE = 0.09;

  function dark() {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function makeDot() {
    return {
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * SPEED,
      vy: (Math.random() - 0.5) * SPEED,
      r: 0.7 + Math.random() * 0.9,
    };
  }

  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Scale the field with the viewport, but keep it sparse on small screens
    // so a phone is not simply a denser version of the same texture.
    var want = Math.round(COUNT * Math.min(1, Math.max(0.45, w / 1200)));
    while (dots.length > want) dots.pop();
    while (dots.length < want) dots.push(makeDot());
  }

  function render(moving) {
    var i, j, a, b, dx, dy, d;
    var fg = dark() ? "255,255,255" : "0,0,0";

    ctx.clearRect(0, 0, w, h);

    for (i = 0; i < dots.length; i++) {
      a = dots[i];

      if (moving) {
        a.x += a.vx;
        a.y += a.vy;

        // Wrap rather than bounce: no edge reversal, so nothing ever pulses
        // against the border.
        if (a.x < -10) a.x = w + 10; else if (a.x > w + 10) a.x = -10;
        if (a.y < -10) a.y = h + 10; else if (a.y > h + 10) a.y = -10;
      }

      for (j = i + 1; j < dots.length; j++) {
        b = dots[j];
        dx = a.x - b.x;
        dy = a.y - b.y;
        d = Math.sqrt(dx * dx + dy * dy);
        if (d < LINK) {
          ctx.strokeStyle = "rgba(" + fg + "," + (ALPHA_LINE * (1 - d / LINK)).toFixed(3) + ")";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }

      ctx.fillStyle = "rgba(" + fg + "," + ALPHA_DOT + ")";
      ctx.beginPath();
      ctx.arc(a.x, a.y, a.r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function step() {
    render(true);
    raf = window.requestAnimationFrame(step);
  }

  function drawStatic() {
    // Reduced motion: one still frame, so the page keeps its texture but
    // nothing moves.
    window.cancelAnimationFrame(raf);
    raf = null;
    render(false);
  }

  function play() {
    if (raf === null && !reduceMotion.matches) raf = window.requestAnimationFrame(step);
  }

  function pause() {
    if (raf !== null) {
      window.cancelAnimationFrame(raf);
      raf = null;
    }
  }

  resize();

  if (reduceMotion.matches) {
    drawStatic();
  } else {
    play();
  }

  // React to the reader changing the setting while the page is open.
  var onMotionChange = function () {
    if (reduceMotion.matches) {
      pause();
      drawStatic();
    } else {
      play();
    }
  };
  if (reduceMotion.addEventListener) reduceMotion.addEventListener("change", onMotionChange);
  else if (reduceMotion.addListener) reduceMotion.addListener(onMotionChange);

  var onResize = function () {
    resize();
    if (reduceMotion.matches) drawStatic();
  };
  window.addEventListener("resize", onResize);

  // No point animating a tab nobody is looking at.
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) pause();
    else play();
  });
})();
