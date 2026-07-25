// Demonstration of Errors data.
// Edit this list to control the "Demonstration of Errors" section:
//   file  = filename inside static/demo/
//   title = card heading
//   gt    = the correct (ground-truth) parameter value        (optional)
//   wrong = the wrong parameter value shown in the gif         (optional)
//   error = our metric's error between gt and wrong            (optional)
// Leave out gt/wrong/error (e.g. for the Ground Truth clip) to hide those lines.
// The array order is the display order. Add a new gif by dropping it in
// static/demo/ and appending an entry here — no need to touch index.html.
window.DEMO_ERROR_ITEMS = [
  {
    file: "GT.gif",
    title: "Ground Truth",
    error: "0.00"
  },
  {
    file: "mobility_0_type_animation.gif",
    title: "Joint Type Error",
    gt: "joint type: revolute",
    wrong: "joint type: prismatic",
    error: "0.00"
  },
  {
    file: "mobility_0_limit_animation.gif",
    title: "Joint Limit Error",
    gt: "joint limit: 0 - 0.76",
    wrong: "joint limit: -0.76 - 0",
    error: "0.00"
  },
  {
    file: "mobility_0_limit2_animation.gif",
    title: "Joint Limit Error 2",
    gt: "joint limit: 0 - 0.76",
    wrong: "joint limit: 0 - 1.20",
    error: "0.00"
  }
];
