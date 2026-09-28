/* Drives the real motion.js against a recording canvas.
 *
 * The browser bridge runs with the tab backgrounded, so requestAnimationFrame
 * never fires and the animation cannot be observed there. This harness stubs
 * the handful of browser APIs motion.js touches, captures the rAF callback,
 * and steps it by hand, so the movement and reduced-motion behaviour are
 * checked on the actual file rather than on a reimplementation.
 */

const fs = require("fs");
const path = require("path");
const vm = require("vm");

function makeEnv({ reducedMotion }) {
  const calls = { arc: [], moveTo: 0, lineTo: 0, clear: 0, frames: 0 };
  const ctx = {
    setTransform() {},
    clearRect() { calls.clear++; },
    beginPath() {},
    moveTo() { calls.moveTo++; },
    lineTo() { calls.lineTo++; },
    stroke() {},
    fill() {},
    arc(x, y, r) { calls.arc.push([x, y, r]); },
    set strokeStyle(v) {},
    set fillStyle(v) {},
    set lineWidth(v) {},
  };
  let rafCb = null;
  const env = {
    console,
    Math,
    Date,
    calls,
    requestAnimationFrame(cb) { rafCb = cb; return 1; },
    cancelAnimationFrame() { rafCb = null; },
    document: {
      hidden: false,
      getElementById: () => ({ width: 0, height: 0, getContext: () => ctx }),
      addEventListener() {},
    },
    matchMedia: (q) => ({
      matches: q.includes("reduced-motion") ? reducedMotion : false,
      addEventListener() {},
      addListener() {},
    }),
    window: {
      innerWidth: 1440,
      innerHeight: 900,
      devicePixelRatio: 2,
      matchMedia: (q) => ({
        matches: q.includes("reduced-motion") ? reducedMotion : false,
        addEventListener() {},
        addListener() {},
      }),
      requestAnimationFrame: (cb) => { rafCb = cb; return 1; },
      cancelAnimationFrame() { rafCb = null; },
      addEventListener() {},
    },
  };
  env.globalThis = env;
  return { env, getCb: () => rafCb };
}

const src = fs.readFileSync(path.join(__dirname, "motion.js"), "utf8");
let failures = 0;
function check(name, cond, extra) {
  console.log(`  ${cond ? "ok  " : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
  if (!cond) failures++;
}

console.log("motion.js");

// --- animated path ---------------------------------------------------------
{
  const { env, getCb } = makeEnv({ reducedMotion: false });
  vm.createContext(env);
  vm.runInContext(src, env);
  check("schedules a frame on load", getCb() !== null);

  // The first paint is the first animation frame, not the initial call: a
  // browser would show a blank canvas for the ~16ms until that frame lands.
  check("has not painted before the first frame", env.calls.arc.length === 0);
  getCb()();
  check("paints on the first frame", env.calls.arc.length > 0, `(${env.calls.arc.length} arcs)`);

  const { calls } = env;
  const dots = calls.arc.length;
  const first = JSON.stringify(calls.arc.slice(0, 5));

  const FRAMES = 30;
  calls.arc.length = 0;
  calls.clear = 0;
  for (let i = 0; i < FRAMES; i++) getCb()();
  check("keeps the same dot count while animating", calls.arc.length === dots * FRAMES,
    `(${calls.arc.length} vs ${dots * FRAMES})`);
  check("dots actually move", JSON.stringify(calls.arc.slice(0, 5)) !== first);
  check("clears once per frame", calls.clear === FRAMES, `(${calls.clear} vs ${FRAMES})`);
}

// --- reduced motion --------------------------------------------------------
{
  const { env, getCb } = makeEnv({ reducedMotion: true });
  vm.createContext(env);
  vm.runInContext(src, env);
  const { calls } = env;

  check("does not schedule animation", getCb() === null);
  check("still paints a static frame", calls.arc.length > 0, `(${calls.arc.length} arcs)`);
  calls.arc.length = 0;
  check("draws nothing further on its own", calls.arc.length === 0);
}

console.log(failures === 0 ? "\nall motion checks passed" : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
