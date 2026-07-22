// Demonstration of Errors data.
// Edit this list to control the "Demonstration of Errors" section:
//   file  = filename inside static/demo/
//   title = card heading
//   gt    = the correct (ground-truth) parameter value   (optional)
//   wrong = the wrong parameter value shown in the gif    (optional)
// Leave out gt/wrong (e.g. for the Ground Truth clip) to show no parameter line.
// The array order is the display order. Add a new gif by dropping it in
// static/demo/ and appending an entry here — no need to touch index.html.
window.DEMO_ERROR_ITEMS = [
  {
    file: "GT.gif",
    title: "Ground Truth"
  },
  {
    file: "mobility_0_type_animation.gif",
    title: "Joint Type Error",
    gt: "revolute",
    wrong: "prismatic"
  },
  {
    file: "mobility_0_limit_animation.gif",
    title: "Joint Limit Error",
    gt: "0° – 90°",
    wrong: "0° – 150°"
  }
];
