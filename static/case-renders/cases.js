// Qualitative-results videos for the project page.
//
// Every clip is a 3-second part-color loop: the object's URDF assigns each link
// a flat color from the viewer's per-part palette, and one loop sweeps every
// movable joint from its resting value to the far end of its own range and
// back. Rendered by
// Resources/ArticulateMetrics/dev_scripts/render_motion/render_motion_video.py.
//
// Files live at static/case-renders/<case id>/<source key>_motion.mp4, so this
// file only lists the cases and the source columns. "joints" is the number of
// movable joints that source's URDF predicts (shown under each clip). A source
// listed in a case's "missing" array renders as an empty cell; delete the entry
// once its clip is rendered.
window.CASE_RENDER_SOURCES = [
  {"key": "GT", "label": "GT"},
  {"key": "SPARK", "label": "SPARK"},
  {"key": "URDFormer", "label": "URDFormer"},
  {"key": "ArtLLM", "label": "ArtLLM"},
  {"key": "ArticulateAnything", "label": "Articulate-Anything"},
  {"key": "Ditto", "label": "Ditto"},
  {"key": "Articraft", "label": "Articraft"},
  {"key": "Particulate", "label": "Particulate"},
  {"key": "Articulate-Anymesh", "label": "Articulate-Anymesh"},
];

window.CASE_RENDER_ITEMS = [
  {
    "id": "3519",
    "category": "Bottle",
    "joints": {
      "GT": 1,
      "SPARK": 1,
      "URDFormer": 1,
      "ArtLLM": 2,
      "ArticulateAnything": 1,
      "Ditto": 1,
      "Articraft": 1,
      "Particulate": 1,
      "Articulate-Anymesh": 1
    },
    "missing": []
  },
  {
    "id": "47645",
    "category": "Box",
    "joints": {
      "GT": 1,
      "SPARK": 1,
      "URDFormer": 1,
      "ArtLLM": 1,
      "ArticulateAnything": 1,
      "Ditto": 1,
      "Articraft": 1,
      "Particulate": 1,
      "Articulate-Anymesh": 2
    },
    "missing": []
  },
  {
    "id": "100520",
    "category": "FoldingChair",
    "joints": {
      "GT": 1,
      "SPARK": 1,
      "URDFormer": 0,
      "ArtLLM": 1,
      "ArticulateAnything": 1,
      "Ditto": 1,
      "Articraft": 2,
      "Particulate": 1,
      "Articulate-Anymesh": 1
    },
    "missing": []
  },
  {
    "id": "101220",
    "category": "Fan",
    "joints": {
      "GT": 1,
      "SPARK": 1,
      "URDFormer": 1,
      "ArtLLM": 3,
      "ArticulateAnything": 1,
      "Ditto": 1,
      "Articraft": 2,
      "Particulate": 1,
      "Articulate-Anymesh": 7
    },
    "missing": []
  }
];
